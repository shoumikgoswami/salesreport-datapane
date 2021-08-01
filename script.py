# # Datapane work

import datapane as dp
from analysis import *

# Enter datapane API token
#datapane_api_token = "" 
#datapane_server_url = "https://datapane.com" 
#dp.login(token=datapane_api_token, server=datapane_server_url)


commentary = 'Supermarket type1 has contributed highest to the overall sales, this can be attributed to the fact that Type1 supermarkets high in numbers and dominate every Tier. However, the business has seen a rapid decline in sales since its launch. In terms of the products being sold, items with low fat content are being sold more due to customer preferences.'

# Enter Heading information

header_image = """
<html>
    <!-- Styling elements of the page -->
    <style type='text/css'>
        #container {
            background: #064e89;
            padding: 2em;
        }
        #container2 {
            background: #e4eaff;
            padding: 25px;
        }
        h1 {
            color:#0b3696;
            text-align:left;
            font-size:50px;
            font-family:verdana;
        }
        h2 {
            color:#ffffff;
            text-align: left;
            display: flex;
            justify-content: space-between;
        }
        span {
            color:#ec4899;
            text-align:left;
            font-size:20px;
        }
        #reportdate {
            color:#000000;
            font-size:15px;
            float:right;
            text-align:right;
            margin-left: 80px;
        }
    </style>
    <div id="container">
    <div id="container2">

        <!-- Enter the company name below -->
        <span><b>ABC</b></span><br>
        <span><b>Company</b></span>

        <!-- Enter the reporting date -->
        <span id = "reportdate"><b>15 JUL 2021 </b></span>

        <!-- Enter the report name -->
        <h1> SALES REPORT </h1>

        <!-- Enter details about what the report is about -->
        <p> This report shows the single pager for the company's sales report. It covers how stores are performing and what all items are selling. Using interactive plots, it all shows how sales have changed over the years. </p>
    </div>
    </div>
</html>

"""

# Enter different heading information

heading1 = """
<html>
    <style type='text/css'>
        #container {
            background: #064e89;
        }
        h2 {
            color:#ffffff;
            text-align: center;
        }
    </style>
    <div id="container">
      <h2> Sales performance across Outlets and Locations </h1>
    </div>
</html>"""

heading2 = """
<html>
    <style type='text/css'>
        #container {
            background: #064e89;
        }
        h2 {
            color:#ffffff;
            text-align: center;
        }
    </style>
    <div id="container">
      <h2> Lifetime sales across business </h1>
    </div>
</html>"""

heading3 = """
<html>
    <style type='text/css'>
        #container {
            background: #064e89;
        }
        h2 {
            color:#ffffff;
            text-align: center;
        }
    </style>
    <div id="container">
      <h2> Number of outlets contributing to sales </h1>
    </div>
</html>"""

heading4 = """
<html>
    <style type='text/css'>
        #container {
            background: #064e89;
        }
        h2 {
            color:#ffffff;
            text-align: center;
        }
    </style>
    <div id="container">
      <h2> Maximum Retail Price of goods sold </h1>
    </div>
</html>"""

heading5 = """
<html>
    <style type='text/css'>
        #container {
            background: #064e89;
        }
        h2 {
            color:#ffffff;
            text-align: center;
        }
    </style>
    <div id="container">
      <h2> Sales based on items and fat content </h1>
    </div>
</html>"""

footer = """
<html>
    <style type='text/css'>
        #container {
            background: #064e89;
        }
        h2 {
            color:#ffffff;
            text-align: center;
        }
    </style>
    <div id="container">
      <h2><br>  </h1>
    </div>
</html>

"""

# Page layout 

page = dp.Group(
        dp.HTML(header_image),
        dp.HTML(heading1),
        dp.Table(table.style.background_gradient(cmap=cm)),
        dp.HTML(heading2),
        dp.Plot(plot5),
        dp.HTML(heading3),
        dp.Group(
            dp.Group(
                dp.Text('''**Sales generated by each outlet type**'''),
                dp.Plot(plot1),
                columns = 1),
            dp.Group(
                dp.Text('''**Number of units for each outlet type**'''),
                dp.Plot(plot3),
                columns = 1),
            columns = 2),
        dp.HTML(heading4),
        dp.Plot(plot2),
        dp.HTML(heading5),
        dp.Group(
            dp.Plot(plot4),
            dp.Group(
                dp.BigNumber(
                     heading="Low Fat sales percentage", 
                     value="64%",
                     change="10%",
                     is_upward_change=True),
                dp.BigNumber(
                     heading="Regular Fat sales percentage", 
                     value="36%",
                     change="5%",
                     is_upward_change=False),
                columns = 1),
            columns = 2),
    dp.Text(commentary),
    dp.HTML(footer),
    columns = 1)


report = dp.Report(page)
#report.save(path='test.html', open=True)
#report.publish(name='Sales report for Bigmart Sales', open=True)