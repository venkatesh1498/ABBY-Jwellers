from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import plotly.graph_objs as go
import uuid, time, os, io, base64, calendar
import qrcode
from prophet import Prophet

DATA_FILE = 'Jewelry_Sales_May_June_2025.xlsx'
OPTIN_FILE = 'optin_customers.xlsx'
app = Flask(__name__)

# ─── Data Loading & Helpers ─────────────────────────────────────────────────────

def load_data():
    df = pd.read_excel(DATA_FILE)
    df.columns = [c.strip() for c in df.columns]
    df['Date of Purchase'] = pd.to_datetime(df['Date of Purchase'], errors='coerce')
    df['Month']     = df['Date of Purchase'].dt.month
    df['MonthName'] = df['Date of Purchase'].dt.strftime('%B')
    df['Year']      = df['Date of Purchase'].dt.year
    return df

def get_dropdowns():
    df = load_data()
    return {
        'Gender':            sorted(df['Gender'].dropna().unique()),
        'Product':           ['Diamond','Gold','Platinum','Silver','Stainless Steel','Other'],
        'Type':              ['Anklets','Bangles','Pendant','Ring','NoseRing','Necklace','Bracelet','Earring','Others'],
        'Customized':        ['Y','N'],
        'Coupon Used':       sorted(df['Coupon Used'].dropna().unique()),
        'Repeat Customer':   sorted(df['Repeat Customer'].dropna().unique()),
        'Product Purchased': ['Y','N'],
    }

def get_summary_numeric(data):
    total_sales      = data['Sales Amount'].sum()
    total_orders     = len(data)
    repeat_customers = data['Repeat Customer'].str.upper().eq('Y').sum()
    conv_rate        = data['Product Purchased'].str.upper().eq('Y').mean() * 100 if total_orders else 0
    avg_rev          = total_sales / total_orders if total_orders else 0
    coupon_used      = data['Coupon Used'].str.upper().eq('Y').sum()
    return {
        'Total Sales':      total_sales,
        'Total Orders':     total_orders,
        'Repeat Customers': repeat_customers,
        'Conversion Rate':  conv_rate,
        'Avg Revenue/Purchase': avg_rev,
        'Coupon Used Count':    coupon_used
    }

def get_summary_formatted(data):
    nums = get_summary_numeric(data)
    return {
        'Total Sales':          f'${nums["Total Sales"]:,.2f}',
        'Total Orders':         nums["Total Orders"],
        'Repeat Customers':     nums["Repeat Customers"],
        'Conversion Rate':      f'{nums["Conversion Rate"]:.1f}%',
        'Avg Revenue/Purchase': f'${nums["Avg Revenue/Purchase"]:,.2f}',
        'Coupon Used Count':    nums["Coupon Used Count"]
    }

def forecast_summary(data):
    ts = data[['Date of Purchase','Sales Amount']].dropna().rename(columns={'Date of Purchase':'ds','Sales Amount':'y'})
    if len(ts) < 2:
        return None, ['Not enough data for forecasting']
    m = Prophet().fit(ts)
    days = 7 if len(ts) < 30 else (30 if len(ts) < 180 else 365)
    future = m.make_future_dataframe(periods=days)
    fc = m.predict(future)
    notes = [
        f'Forecasting next {days} days',
        f'Avg Daily Sales ≈ ${fc["yhat"].tail(days).mean():,.2f}'
    ]
    return fc, notes

def generate_charts(data, chart_type):
    charts = []
    def dist_plot(col, title):
        cnt = data[col].value_counts()
        if cnt.empty: return
        fig = go.Figure()
        if chart_type=='bar':
            fig.add_trace(go.Bar(x=cnt.index, y=cnt.values))
        elif chart_type=='line':
            fig.add_trace(go.Scatter(x=cnt.index, y=cnt.values, mode='lines+markers'))
        else:
            fig.add_trace(go.Pie(labels=cnt.index, values=cnt.values))
        fig.update_layout(title=title, height=400)
        charts.append({'id': uuid.uuid4().hex, 'title': title, 'chart': fig.to_html(full_html=False)})
    for col,title in [
        ('Product','Sales by Product'),
        ('Type','Jewelry Type Distribution'),
        ('Customized','Customized vs Non-Customized'),
        ('Coupon Used','Coupon Usage'),
        ('Gender','Gender Breakdown'),
        ('Product Purchased','Purchase Status'),
    ]:
        dist_plot(col, title)
    # Weekly
    data['Week'] = data['Date of Purchase'].dt.to_period('W').apply(lambda r: r.start_time)
    weekly = data[data['Product Purchased']=='Y'].groupby('Week')['Sales Amount'].sum().reset_index()
    fig = go.Figure([go.Bar(x=weekly['Week'], y=weekly['Sales Amount'])])
    fig.update_layout(title='Weekly Sales Trend', height=400)
    charts.append({'id':uuid.uuid4().hex,'title':'Weekly Sales Trend','chart':fig.to_html(full_html=False)})
    # Top 10
    top = data.groupby('Customer Name')['Sales Amount'].sum().nlargest(10).reset_index()
    fig = go.Figure([go.Bar(x=top['Customer Name'], y=top['Sales Amount'])])
    fig.update_layout(title='Top 10 Customers', height=400)
    charts.append({'id':uuid.uuid4().hex,'title':'Top 10 Customers','chart':fig.to_html(full_html=False)})
    # Forecast
    fc, notes = forecast_summary(data)
    if fc is not None:
        fig = go.Figure([
            go.Scatter(x=fc['ds'], y=fc['yhat'], name='Forecast'),
            go.Scatter(x=fc['ds'], y=fc['yhat_lower'], name='Lower', line={'dash':'dot'}),
            go.Scatter(x=fc['ds'], y=fc['yhat_upper'], name='Upper', line={'dash':'dot'})
        ])
        fig.update_layout(title='Forecast - Future Sales', height=400)
        charts.append({'id':uuid.uuid4().hex,'title':'Forecast - Future Sales','chart':fig.to_html(full_html=False)})
    return charts, notes

def compute_insights(df):
    df = df.dropna(subset=['Gender','Product Purchased','Sales Amount','Date of Purchase'])
    pr = df.groupby('Gender')['Product Purchased'].apply(lambda x: x.eq('Y').mean())
    m_rate, f_rate = pr.get('M',0), pr.get('F',0)
    insights = []
    if f_rate>m_rate:
        diff=f_rate-m_rate
        insights.append(f"Female customers purchase at {f_rate:.1%} vs male at {m_rate:.1%}, a {diff:.1%} higher rate.")
    else:
        diff=m_rate-f_rate
        insights.append(f"Male customers purchase at {m_rate:.1%} vs female at {f_rate:.1%}, a {diff:.1%} higher rate.")
    df['Day']=df['Date of Purchase'].dt.day_name()
    best_day = df.groupby('Day')['Sales Amount'].sum().idxmax()
    insights.append(f"The best day for sales is **{best_day}**.")
    top_f = df[df['Gender']=='F'].groupby('Product')['Sales Amount'].sum().idxmax()
    top_m = df[df['Gender']=='M'].groupby('Product')['Sales Amount'].sum().idxmax()
    insights.append(f"Females buy **{top_f}** most, males buy **{top_m}** most.")
    insights.append("Consider targeted promotions on your best day and tailor product highlights by gender.")
    return insights

def make_qr_code():
    target = url_for('qr_form', _external=True)
    img = qrcode.make(target)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()

# ─── ROUTES ─────────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET','POST'])
def home():
    df       = load_data()
    insights = compute_insights(df)
    months   = sorted(df['MonthName'].unique(),
                      key=lambda m: list(calendar.month_name).index(m))
    years    = sorted(df['Year'].unique())

    mode       = request.form.get('mode','all')
    sel_months = request.form.getlist('month')
    sel_years  = request.form.getlist('year')
    chart_type = request.form.get('chart_type','bar')

    # build df_list
    if mode=='single' and sel_months and sel_years:
        label  = f"{sel_months[0]} {sel_years[0]}"
        subset = df[(df['MonthName']==sel_months[0]) & (df['Year']==int(sel_years[0]))]
        df_list = [(label, subset)]
    elif mode=='compare' and (sel_months or sel_years):
        df_list=[]
        if sel_months and not sel_years:
            for m in sel_months:
                df_list.append((m, df[df['MonthName']==m]))
        elif sel_years and not sel_months:
            for y in sel_years:
                df_list.append((str(y), df[df['Year']==int(y)]))
        else:
            for m in sel_months:
                for y in sel_years:
                    lab = f"{m} {y}"
                    df_list.append((lab, df[(df['MonthName']==m)&(df['Year']==int(y))]))
    else:
        df_list = [('All', df)]

    # assemble periods
    periods=[]
    for label, subset in df_list:
        fmt = get_summary_formatted(subset)
        # always wrap with change="N/A"
        wrapped = {k:{'value':v, 'change':'N/A'} for k,v in fmt.items()}
        periods.append({'label':label,'summary':wrapped,'charts': generate_charts(subset,chart_type)[0]})

    # compute real %change for single mode
    if mode=='single' and df_list and df_list[0][0]!='All':
        label, subset = df_list[0]
        m_sel,y_sel= label.split()
        m_idx=list(calendar.month_name).index(m_sel)
        prev_m=12 if m_idx==1 else m_idx-1
        prev_y=int(y_sel)-1 if m_idx==1 else int(y_sel)
        prev_df=df[(df['Month']==prev_m)&(df['Year']==prev_y)]
        curr_nums=get_summary_numeric(subset)
        prev_nums=get_summary_numeric(prev_df)
        for k in periods[0]['summary']:
            prev=prev_nums.get(k,0) or 0
            curr=curr_nums.get(k,0)
            if prev:
                pct=(curr-prev)/prev*100
                periods[0]['summary'][k]['change']=f"{pct:+.1f}%"
            else:
                periods[0]['summary'][k]['change']="N/A"

    # build compare charts
    compare_chart      = None
    compare_dist_charts=[]
    if mode=='compare' and len(df_list)>1:
        fig=go.Figure()
        for label, sub in df_list:
            wk=sub.copy()
            wk['Week']=wk['Date of Purchase'].dt.to_period('W').apply(lambda r:r.start_time)
            agg=wk[wk['Product Purchased']=='Y'].groupby('Week')['Sales Amount'].sum().reset_index()
            fig.add_trace(go.Scatter(x=agg['Week'],y=agg['Sales Amount'],mode='lines+markers',name=label))
        fig.update_layout(title='Weekly Sales Comparison',height=450)
        compare_chart=fig.to_html(full_html=False)
        for col,title in [
            ('Product','Sales by Product'),
            ('Type','Jewelry Type Distribution'),
            ('Customized','Customized vs Non-Customized'),
            ('Coupon Used','Coupon Usage'),
            ('Gender','Gender Breakdown'),
            ('Product Purchased','Purchase Status'),
        ]:
            fig2=go.Figure()
            for label, sub in df_list:
                cnt=sub[col].value_counts()
                fig2.add_trace(go.Bar(x=cnt.index,y=cnt.values,name=label))
            fig2.update_layout(title=f'Comparison: {title}',barmode='group',height=400)
            compare_dist_charts.append({'id':uuid.uuid4().hex,'title':title,'chart':fig2.to_html(full_html=False)})

    return render_template(
        'dashboard_06302025_QRCode_integrated.html',
        view='dashboard', mode=mode,
        months=months, years=years,
        selected_months=sel_months,
        selected_years=sel_years,
        chart_type=chart_type,
        dropdowns=get_dropdowns(),
        periods=periods,
        compare_chart=compare_chart,
        compare_dist_charts=compare_dist_charts,
        insights=insights,
        qr_code=make_qr_code()
    )

@app.route('/order', methods=['GET','POST'])
def order():
    dropdowns=get_dropdowns()
    if request.method=='POST':
        f=request.form
        amount=float(f.get('amount') or 0) if f.get('product_purchased')=='Y' else 0
        row={
            'Customer Name':     f.get('customer_name','').strip(),
            'Mobile Number':     f.get('mobile','').strip(),
            'Email Address':     f.get('email','').strip(),
            'Gender':            f.get('gender',''),
            'Product':           f.get('product',''),
            'Type':              f.get('type',''),
            'Customized':        f.get('customized',''),
            'Coupon Used':       f.get('coupon_used',''),
            'Repeat Customer':   f.get('repeat_customer',''),
            'Product Purchased': f.get('product_purchased',''),
            'Sales Amount':      amount,
            'Date of Purchase':  pd.to_datetime(f.get('date'))
        }
        df=load_data()
        df=pd.concat([df,pd.DataFrame([row])],ignore_index=True)
        df.to_excel(DATA_FILE,index=False)
        time.sleep(0.5)
        return redirect(url_for('home'))
    return render_template(
        'dashboard_06302025_QRCode_integrated.html',
        view='order',
        dropdowns=dropdowns,
        qr_code=make_qr_code()
    )

@app.route('/qr_form', methods=['GET','POST'])
def qr_form():
    if request.method=='POST':
        f=request.form
        rec={
            'Customer Name': f.get('customer_name','').strip(),
            'Mobile Number': f.get('mobile','').strip(),
            'BirthDate':     f.get('birthdate',''),
            'Opt-In':        'Y' if f.get('optin')=='on' else 'N'
        }
        if os.path.exists(OPTIN_FILE):
            odf=pd.read_excel(OPTIN_FILE)
            odf=pd.concat([odf,pd.DataFrame([rec])],ignore_index=True)
        else:
            odf=pd.DataFrame([rec])
        odf.to_excel(OPTIN_FILE,index=False)
        return render_template('qr_submitted.html', name=rec['Customer Name'])
    return render_template('qr_form.html')

if __name__=='__main__':
    app.run(debug=True)
