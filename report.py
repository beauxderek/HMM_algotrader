import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime
from reportlab.pdfgen import canvas

def generate_pairs_trading_report(report_data: dict, filename: str):
    """Generate comprehensive report for pairs trading strategy"""
    
    # Create PDF document
    doc = SimpleDocTemplate(filename, pagesize=landscape(letter))
    elements = []
    styles = getSampleStyleSheet()
    
    # Add title
    elements.append(Paragraph("Pairs Trading Strategy Performance Report", styles['Title']))
    elements.append(Spacer(1, 20))
    
    # Add overview section
    elements.append(Paragraph("Strategy Overview", styles['Heading1']))
    overview_data = [
        ["Parameter", "Value"],
        ["Time Period", f"{report_data['start_date']} to {report_data['end_date']}"],
        ["Initial Capital", f"${report_data['initial_capital']:,.2f}"],
        ["Final Capital", f"${report_data['final_capital']:,.2f}"],
        ["Number of Pairs", f"{len(report_data['pairs'])}"],
        ["Timeframe", report_data['timeframe']]
    ]
    
    elements.append(Table(overview_data, style=TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ])))
    elements.append(Spacer(1, 20))
    
    # Overall performance metrics
    elements.append(Paragraph("Overall Performance", styles['Heading1']))
    metrics_table = _create_metrics_table(report_data['metrics'])
    elements.append(metrics_table)
    elements.append(Spacer(1, 20))
    
    # Portfolio value chart
    elements.append(Paragraph("Portfolio Performance", styles['Heading1']))
    portfolio_chart = _create_portfolio_chart(report_data)
    elements.append(Image(portfolio_chart, width=8*inch, height=4*inch))
    elements.append(PageBreak())
    
    # Individual pair analysis
    for pair in report_data['pairs']:
        pair_str = f"{pair[0]}-{pair[1]}"
        elements.append(Paragraph(f"Pair Analysis: {pair_str}", styles['Heading1']))
        
        # Pair metrics
        pair_metrics = report_data['pair_metrics'][pair]
        elements.append(Paragraph("Pair Performance Metrics", styles['Heading2']))
        elements.append(_create_metrics_table(pair_metrics))
        elements.append(Spacer(1, 20))
        
        # Spread and z-score analysis
        spread_chart = _create_spread_chart(report_data, pair)
        elements.append(Image(spread_chart, width=8*inch, height=4*inch))
        elements.append(Spacer(1, 20))
        
        # Trading signals visualization
        signals_chart = _create_signals_chart(report_data, pair)
        elements.append(Image(signals_chart, width=8*inch, height=4*inch))
        elements.append(PageBreak())
    
    # Correlation matrix for multiple pairs
    if len(report_data['pairs']) > 1:
        elements.append(Paragraph("Pairs Correlation Analysis", styles['Heading1']))
        corr_chart = _create_correlation_chart(report_data)
        elements.append(Image(corr_chart, width=7*inch, height=7*inch))
        elements.append(PageBreak())
    
    # Build the PDF
    doc.build(elements)

def _create_metrics_table(metrics: dict) -> Table:
    """Create formatted metrics table"""
    data = [["Metric", "Value"]]
    for key, value in metrics.items():
        formatted_key = key.replace('_', ' ').title()
        if isinstance(value, float):
            if 'return' in key or 'drawdown' in key:
                formatted_value = f"{value:.2%}"
            else:
                formatted_value = f"{value:.2f}"
        else:
            formatted_value = str(value)
        data.append([formatted_key, formatted_value])
    
    return Table(data, style=TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ]))

def _create_portfolio_chart(report_data: dict) -> BytesIO:
    """Create portfolio value chart"""
    plt.figure(figsize=(10, 6))
    plt.plot(report_data['portfolio_values'], label='Portfolio Value')
    plt.title('Portfolio Performance')
    plt.xlabel('Time')
    plt.ylabel('Value ($)')
    plt.grid(True)
    plt.legend()
    
    img_stream = BytesIO()
    plt.savefig(img_stream, format='png', bbox_inches='tight')
    plt.close()
    img_stream.seek(0)
    return img_stream

def _create_spread_chart(report_data: dict, pair: tuple) -> BytesIO:
    """Create spread and z-score analysis chart"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot spread
    ax1.plot(report_data['spreads'][pair], label='Spread')
    ax1.set_title(f'Price Spread: {pair[0]}-{pair[1]}')
    ax1.grid(True)
    ax1.legend()
    
    # Plot z-score
    ax2.plot(report_data['zscores'][pair], label='Z-Score')
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.axhline(y=2, color='g', linestyle='--')
    ax2.axhline(y=-2, color='g', linestyle='--')
    ax2.set_title('Z-Score')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    img_stream = BytesIO()
    plt.savefig(img_stream, format='png', bbox_inches='tight')
    plt.close()
    img_stream.seek(0)
    return img_stream

def _create_signals_chart(report_data: dict, pair: tuple) -> BytesIO:
    """Create trading signals visualization"""
    signals = pd.DataFrame(report_data['signals_history'][pair])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot positions
    ax1.plot(signals['position1'], label=f'{pair[0]} Position')
    ax1.plot(signals['position2'], label=f'{pair[1]} Position')
    ax1.set_title('Position Sizes')
    ax1.grid(True)
    ax1.legend()
    
    # Plot hedge ratio
    ax2.plot(report_data['hedge_ratios'][pair], label='Hedge Ratio')
    ax2.set_title('Dynamic Hedge Ratio')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    img_stream = BytesIO()
    plt.savefig(img_stream, format='png', bbox_inches='tight')
    plt.close()
    img_stream.seek(0)
    return img_stream

def _create_correlation_chart(report_data: dict) -> BytesIO:
    """Create correlation matrix for pair returns"""
    # Create returns matrix
    pair_returns = pd.DataFrame({
        f"{p[0]}-{p[1]}": report_data['pair_returns'][p]
        for p in report_data['pairs']
    })
    
    # Calculate correlation matrix
    corr_matrix = pair_returns.corr()
    
    # Create heatmap
    plt.figure(figsize=(8, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Pair Returns Correlation Matrix')
    
    img_stream = BytesIO()
    plt.savefig(img_stream, format='png', bbox_inches='tight')
    plt.close()
    img_stream.seek(0)
    return img_stream

def generate_trend_following_report(report_data: dict, filename: str):
    """Generate comprehensive report for trend following strategy"""
    
    # Create PDF document
    doc = SimpleDocTemplate(filename, pagesize=landscape(letter))
    elements = []
    styles = getSampleStyleSheet()
    
    # Add title
    elements.append(Paragraph("HMM Trend Following Strategy Performance Report", styles['Title']))
    elements.append(Spacer(1, 20))
    
    # Add overview section
    elements.append(Paragraph("Strategy Overview", styles['Heading1']))
    overview_data = [
        ["Parameter", "Value"],
        ["Time Period", f"{report_data['start_date']} to {report_data['end_date']}"],
        ["Initial Capital", f"${report_data['initial_capital']:,.2f}"],
        ["Final Capital", f"${report_data['final_capital']:,.2f}"],
        ["Number of Symbols", f"{len(report_data['symbols'])}"],
        ["Timeframe", report_data['timeframe']]
    ]
    
    elements.append(Table(overview_data, style=TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ])))
    elements.append(Spacer(1, 20))
    
    # Overall performance metrics
    elements.append(Paragraph("Overall Performance", styles['Heading1']))
    metrics_table = _create_metrics_table(report_data['metrics'])
    elements.append(metrics_table)
    elements.append(Spacer(1, 20))
    
    # Portfolio value chart
    elements.append(Paragraph("Portfolio Performance", styles['Heading1']))
    portfolio_chart = _create_portfolio_chart(report_data)
    elements.append(Image(portfolio_chart, width=8*inch, height=4*inch))
    elements.append(PageBreak())
    
    # Individual symbol analysis
    for symbol in report_data['symbols']:
        elements.append(Paragraph(f"Symbol Analysis: {symbol}", styles['Heading1']))
        
        # Symbol performance metrics
        signals = pd.DataFrame(report_data['signals_history'][symbol])
        
        # Create state probability chart
        prob_chart = _create_probability_chart(signals, symbol)
        elements.append(Image(prob_chart, width=8*inch, height=4*inch))
        elements.append(Spacer(1, 20))
        
        # Create trading signals chart
        signals_chart = _create_trading_signals_chart(signals, symbol)
        elements.append(Image(signals_chart, width=8*inch, height=4*inch))
        elements.append(PageBreak())
    
    # Trade analysis
    elements.append(Paragraph("Trade Analysis", styles['Heading1']))
    trades_df = pd.DataFrame(report_data['trades'])
    if not trades_df.empty:
        trade_analysis = _create_trade_analysis_chart(trades_df)
        elements.append(Image(trade_analysis, width=8*inch, height=4*inch))
    
    # Build the PDF
    doc.build(elements)
    print(f"Trend following report generated: {filename}")

def _create_metrics_table(metrics: dict) -> Table:
    """Create formatted metrics table"""
    data = [["Metric", "Value"]]
    for key, value in metrics.items():
        formatted_key = key.replace('_', ' ').title()
        if isinstance(value, float):
            if 'return' in key or 'drawdown' in key:
                formatted_value = f"{value:.2%}"
            else:
                formatted_value = f"{value:.2f}"
        else:
            formatted_value = str(value)
        data.append([formatted_key, formatted_value])
    
    return Table(data, style=TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ]))

def _create_portfolio_chart(report_data: dict) -> BytesIO:
    """Create portfolio value chart"""
    plt.figure(figsize=(10, 6))
    plt.plot(report_data['portfolio_values'], label='Portfolio Value')
    plt.title('Portfolio Performance')
    plt.xlabel('Time')
    plt.ylabel('Value ($)')
    plt.grid(True)
    plt.legend()
    
    img_stream = BytesIO()
    plt.savefig(img_stream, format='png', bbox_inches='tight')
    plt.close()
    img_stream.seek(0)
    return img_stream

def _create_probability_chart(signals: pd.DataFrame, symbol: str) -> BytesIO:
    """Create state probability visualization"""
    plt.figure(figsize=(10, 6))
    plt.plot(signals.index, signals['bull_prob'], label='Bull Probability', color='green')
    plt.plot(signals.index, 1 - signals['bull_prob'], label='Bear Probability', color='red')
    plt.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=0.3, color='gray', linestyle='--', alpha=0.5)
    plt.title(f'State Probabilities - {symbol}')
    plt.xlabel('Time')
    plt.ylabel('Probability')
    plt.grid(True)
    plt.legend()
    
    img_stream = BytesIO()
    plt.savefig(img_stream, format='png', bbox_inches='tight')
    plt.close()
    img_stream.seek(0)
    return img_stream

def _create_trading_signals_chart(signals: pd.DataFrame, symbol: str) -> BytesIO:
    """Create trading signals visualization"""
    plt.figure(figsize=(10, 6))
    plt.plot(signals.index, signals['position'], label='Position Size', color='blue')
    plt.plot(signals.index, signals['confidence'], label='Signal Confidence', color='orange', alpha=0.5)
    plt.title(f'Trading Signals - {symbol}')
    plt.xlabel('Time')
    plt.ylabel('Position Size / Confidence')
    plt.grid(True)
    plt.legend()
    
    img_stream = BytesIO()
    plt.savefig(img_stream, format='png', bbox_inches='tight')
    plt.close()
    img_stream.seek(0)
    return img_stream

def _create_trade_analysis_chart(trades_df: pd.DataFrame) -> BytesIO:
    """Create trade analysis visualization"""
    plt.figure(figsize=(10, 6))
    
    # Plot cumulative P&L
    trades_df['cumulative_value'] = trades_df['value'].cumsum()
    plt.plot(trades_df.index, trades_df['cumulative_value'], label='Cumulative P&L')
    
    # Mark trades
    plt.scatter(trades_df[trades_df['value'] > 0].index, 
               trades_df[trades_df['value'] > 0]['cumulative_value'],
               color='green', alpha=0.5, label='Winning Trades')
    plt.scatter(trades_df[trades_df['value'] < 0].index,
               trades_df[trades_df['value'] < 0]['cumulative_value'],
               color='red', alpha=0.5, label='Losing Trades')
    
    plt.title('Trade Analysis')
    plt.xlabel('Trade Number')
    plt.ylabel('Cumulative P&L ($)')
    plt.grid(True)
    plt.legend()
    
    img_stream = BytesIO()
    plt.savefig(img_stream, format='png', bbox_inches='tight')
    plt.close()
    img_stream.seek(0)
    return img_stream

# The pairs trading report function remains as previously shared
def generate_pairs_trading_report(report_data: dict, filename: str):
    """Generate comprehensive report for pairs trading strategy"""
    # [Previous implementation stays the same]
    pass