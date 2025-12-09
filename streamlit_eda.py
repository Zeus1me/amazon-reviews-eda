"""
Amazon Movie Reviews - EDA Dashboard
=====================================
Cloud-ready Streamlit Dashboard

To run locally: streamlit run app.py
"""

import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ========= PAGE CONFIG =========
st.set_page_config(
    page_title="Amazon Reviews EDA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========= CHART COLORS =========
CHART_COLORS = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
                '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']


# ========= DATA LOADING =========
@st.cache_data(ttl=3600)
def load_data_from_file(uploaded_file):
    """Load data from uploaded file."""
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.parquet'):
        return pd.read_parquet(uploaded_file)
    else:
        st.error("Please upload a CSV or Parquet file.")
        return None


@st.cache_data(ttl=3600)
def load_sample_data():
    """Load sample data for demo purposes."""
    # Create sample data for demonstration
    np.random.seed(42)
    n = 10000

    years = np.random.choice(range(2005, 2013), n, p=[0.05, 0.08, 0.10, 0.12, 0.15, 0.15, 0.17, 0.18])
    ratings = np.random.choice([1, 2, 3, 4, 5], n, p=[0.08, 0.05, 0.09, 0.20, 0.58])

    df = pd.DataFrame({
        'productId': [f'B00{np.random.randint(10000, 99999)}' for _ in range(n)],
        'userId': [f'A{np.random.randint(100000, 999999)}' for _ in range(n)],
        'rating': ratings,
        'review_year': years,
        'helpful_yes': np.random.randint(0, 50, n),
        'total_votes': np.random.randint(0, 100, n),
    })

    df['helpful_ratio'] = np.where(df['total_votes'] > 0, df['helpful_yes'] / df['total_votes'], 0)
    df['review_date'] = pd.to_datetime(df['review_year'].astype(str) + '-' +
                                       np.random.randint(1, 13, n).astype(str) + '-' +
                                       np.random.randint(1, 28, n).astype(str))

    return df


@st.cache_data(ttl=3600)
def load_local_data(folder_path):
    """Load data from local parquet files."""
    folder = Path(folder_path)
    if not folder.exists():
        return None

    parquet_files = list(folder.glob("*.parquet"))
    if not parquet_files:
        return None

    dfs = [pd.read_parquet(pf) for pf in parquet_files]
    return pd.concat(dfs, ignore_index=True)


# ========= HELPERS =========
def format_number(num):
    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    return f"{num:,.0f}"


def get_filtered_data(df, filters):
    filtered = df.copy()
    if filters.get('year_range'):
        filtered = filtered[
            (filtered['review_year'] >= filters['year_range'][0]) &
            (filtered['review_year'] <= filters['year_range'][1])
            ]
    if filters.get('rating_filter') and len(filters['rating_filter']) > 0:
        filtered = filtered[filtered['rating'].isin(filters['rating_filter'])]
    if filters.get('min_votes'):
        filtered = filtered[filtered['total_votes'] >= filters['min_votes']]
    return filtered


# ========= SIDEBAR =========
def render_sidebar(df):
    st.sidebar.title("Amazon Reviews EDA")
    st.sidebar.caption("Interactive Dashboard")
    st.sidebar.divider()

    # Dataset Summary
    st.sidebar.subheader("Dataset Summary")
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Reviews", format_number(len(df)))
    col2.metric("Users", format_number(df['userId'].nunique()))
    col1.metric("Products", format_number(df['productId'].nunique()))
    col2.metric("Avg Rating", f"{df['rating'].mean():.2f}")

    st.sidebar.divider()

    # Filters
    st.sidebar.subheader("Filters")

    year_min, year_max = int(df['review_year'].min()), int(df['review_year'].max())
    year_range = st.sidebar.slider("Year Range", year_min, year_max, (year_min, year_max))

    rating_filter = st.sidebar.multiselect("Ratings", options=[1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])

    min_votes = st.sidebar.slider("Minimum Votes", 0, 50, 0)

    filters = {'year_range': year_range, 'rating_filter': rating_filter, 'min_votes': min_votes}

    filtered_df = get_filtered_data(df, filters)
    st.sidebar.info(f"Showing {format_number(len(filtered_df))} reviews ({len(filtered_df) / len(df) * 100:.1f}%)")

    st.sidebar.divider()

    # Navigation
    st.sidebar.subheader("Navigation")
    section = st.sidebar.radio(
        "Select Section",
        ["Overview", "Time Trends", "Rating Analysis", "Reviewer Analysis", "Helpfulness Analysis", "Product Analysis"],
        label_visibility="collapsed"
    )

    st.sidebar.divider()

    # Export
    st.sidebar.subheader("Export")
    csv_data = filtered_df.to_csv(index=False)
    st.sidebar.download_button("Download Filtered Data (CSV)", csv_data, "filtered_reviews.csv", "text/csv")

    st.sidebar.divider()
    st.sidebar.caption("ALY6110 Final Project | Zeus | 2024")

    return section, filters


# ========= OVERVIEW =========
def render_overview(df, filters):
    filtered_df = get_filtered_data(df, filters)

    st.title("Amazon Movie Reviews Dashboard")
    st.caption("Interactive Exploratory Data Analysis")
    st.divider()

    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Reviews", format_number(len(filtered_df)))
    col2.metric("Unique Users", format_number(filtered_df['userId'].nunique()))
    col3.metric("Unique Products", format_number(filtered_df['productId'].nunique()))
    col4.metric("Avg Rating", f"{filtered_df['rating'].mean():.2f}")
    col5.metric("Year Range", f"{int(filtered_df['review_year'].min())}-{int(filtered_df['review_year'].max())}")

    st.divider()

    # Charts
    tab1, tab2, tab3 = st.tabs(["Rating Overview", "Time Overview", "Quick Stats"])

    with tab1:
        col1, col2 = st.columns([3, 2])

        with col1:
            rating_counts = filtered_df['rating'].value_counts().sort_index()
            fig = go.Figure(data=[go.Bar(
                x=[f"{int(r)} Stars" for r in rating_counts.index],
                y=rating_counts.values,
                marker_color=CHART_COLORS[:5],
                text=[format_number(v) for v in rating_counts.values],
                textposition='outside'
            )])
            fig.update_layout(title="Rating Distribution", xaxis_title="Rating", yaxis_title="Count", height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure(data=[go.Pie(
                labels=[f'{int(r)} Stars' for r in rating_counts.index],
                values=rating_counts.values,
                hole=0.4,
                marker_colors=CHART_COLORS[:5]
            )])
            fig.update_layout(title="Rating Share", height=400)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        yearly = filtered_df.groupby('review_year').agg(count=('rating', 'count'),
                                                        avg_rating=('rating', 'mean')).reset_index()

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=yearly['review_year'], y=yearly['count'], name="Review Count",
                             marker_color=CHART_COLORS[0], opacity=0.7), secondary_y=False)
        fig.add_trace(go.Scatter(x=yearly['review_year'], y=yearly['avg_rating'], name="Avg Rating",
                                 mode='lines+markers', marker=dict(color=CHART_COLORS[1], size=8),
                                 line=dict(width=3)), secondary_y=True)
        fig.update_layout(title="Review Volume and Rating Over Time", height=400, hovermode='x unified')
        fig.update_yaxes(title_text="Review Count", secondary_y=False)
        fig.update_yaxes(title_text="Average Rating", secondary_y=True, range=[3.5, 5])
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Volume Statistics**")
            rpu = filtered_df.groupby('userId').size()
            rpp = filtered_df.groupby('productId').size()
            stats = pd.DataFrame({
                'Metric': ['Avg Reviews/User', 'Median Reviews/User', 'Avg Reviews/Product', 'Median Reviews/Product'],
                'Value': [f"{rpu.mean():.1f}", f"{rpu.median():.0f}", f"{rpp.mean():.1f}", f"{rpp.median():.0f}"]
            })
            st.dataframe(stats, hide_index=True, use_container_width=True)

        with col2:
            st.markdown("**Rating Breakdown**")
            pos = (filtered_df['rating'] >= 4).mean() * 100
            neu = (filtered_df['rating'] == 3).mean() * 100
            neg = (filtered_df['rating'] <= 2).mean() * 100
            fig = go.Figure(data=[go.Pie(
                labels=['Positive (4-5)', 'Neutral (3)', 'Negative (1-2)'],
                values=[pos, neu, neg],
                marker_colors=[CHART_COLORS[2], CHART_COLORS[4], CHART_COLORS[1]],
                hole=0.4
            )])
            fig.update_layout(height=250, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            st.markdown("**Helpfulness Stats**")
            if 'total_votes' in filtered_df.columns:
                voted = filtered_df[filtered_df['total_votes'] > 0]
                st.metric("Reviews with Votes", format_number(len(voted)))
                if 'helpful_ratio' in voted.columns and len(voted) > 0:
                    st.metric("Avg Helpful Ratio", f"{voted['helpful_ratio'].mean():.1%}")
                    st.metric("Highly Helpful (80%+)", format_number((voted['helpful_ratio'] >= 0.8).sum()))

    # Summary
    st.divider()
    pos_share = (filtered_df['rating'] >= 4).mean() * 100
    neg_share = (filtered_df['rating'] <= 2).mean() * 100
    st.info(
        f"**Summary**: {format_number(len(filtered_df))} reviews from {format_number(filtered_df['userId'].nunique())} users across {format_number(filtered_df['productId'].nunique())} products. {pos_share:.1f}% positive, {neg_share:.1f}% negative. Average: {filtered_df['rating'].mean():.2f} stars.")

    # Data Preview
    st.divider()
    st.subheader("Data Preview")
    col1, col2 = st.columns([3, 1])
    with col1:
        search = st.text_input("Search by Product or User ID", placeholder="Enter ID...")
    with col2:
        n_rows = st.selectbox("Rows", [10, 25, 50, 100], index=0)

    display_df = filtered_df.copy()
    if search:
        display_df = display_df[
            display_df['productId'].str.contains(search, case=False, na=False) |
            display_df['userId'].str.contains(search, case=False, na=False)
            ]
    st.dataframe(display_df.head(n_rows), use_container_width=True)


# ========= TIME TRENDS =========
def render_time_trends(df, filters):
    filtered_df = get_filtered_data(df, filters)

    st.title("Time Trends Analysis")
    st.divider()

    chart_type = st.selectbox("Chart Type", ["Bar + Line", "Area", "Line"])

    yearly = filtered_df.groupby('review_year').agg(count=('rating', 'count'),
                                                    avg_rating=('rating', 'mean')).reset_index()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews", format_number(yearly['count'].sum()))
    peak_idx = yearly['count'].idxmax()
    col2.metric("Peak Year", int(yearly.loc[peak_idx, 'review_year']))
    col3.metric("Peak Volume", format_number(yearly['count'].max()))
    growth = ((yearly['count'].iloc[-1] / yearly['count'].iloc[0]) - 1) * 100 if len(yearly) > 1 else 0
    col4.metric("Growth", f"{growth:+.1f}%")

    st.divider()

    if chart_type == "Bar + Line":
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=yearly['review_year'], y=yearly['count'], name="Reviews", marker_color=CHART_COLORS[0],
                             opacity=0.7), secondary_y=False)
        fig.add_trace(
            go.Scatter(x=yearly['review_year'], y=yearly['avg_rating'], name="Avg Rating", mode='lines+markers',
                       line=dict(color=CHART_COLORS[1], width=3)), secondary_y=True)
        fig.update_yaxes(title_text="Count", secondary_y=False)
        fig.update_yaxes(title_text="Avg Rating", secondary_y=True, range=[3.5, 5])
    elif chart_type == "Area":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=yearly['review_year'], y=yearly['count'], fill='tozeroy', mode='lines',
                                 line=dict(color=CHART_COLORS[0]), name="Reviews"))
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=yearly['review_year'], y=yearly['count'], mode='lines+markers',
                                 line=dict(color=CHART_COLORS[0], width=3), name="Reviews"))

    fig.update_layout(title="Review Trends", height=450, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Rating Distribution by Year")

    rating_year = filtered_df.groupby(['review_year', 'rating']).size().unstack(fill_value=0)
    rating_year_pct = rating_year.div(rating_year.sum(axis=1), axis=0) * 100

    fig = go.Figure(data=go.Heatmap(
        z=rating_year_pct.T.values,
        x=rating_year_pct.index.astype(int),
        y=[f'{int(r)} Stars' for r in rating_year_pct.columns],
        colorscale='Blues',
        text=rating_year_pct.T.round(1).values,
        texttemplate='%{text}%'
    ))
    fig.update_layout(title="Rating Share by Year (%)", height=300, xaxis_title="Year", yaxis_title="Rating")
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"**Insights**: Volume changed by {growth:+.1f}%. Peak: {int(yearly.loc[peak_idx, 'review_year'])} ({format_number(yearly['count'].max())} reviews).")


# ========= RATING ANALYSIS =========
def render_rating_analysis(df, filters):
    filtered_df = get_filtered_data(df, filters)

    st.title("Rating Analysis")
    st.divider()

    ratings = filtered_df['rating'].dropna()
    counts = ratings.value_counts().sort_index()
    total = counts.sum()
    pcts = (counts / total * 100).round(1)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Average", f"{ratings.mean():.2f}")
    col2.metric("Median", f"{ratings.median():.0f}")
    col3.metric("Mode", f"{ratings.mode().iloc[0]:.0f}")
    col4.metric("Positive (4-5)", f"{pcts[pcts.index >= 4].sum():.1f}%")
    col5.metric("Negative (1-2)", f"{pcts[pcts.index <= 2].sum():.1f}%")

    st.divider()

    tab1, tab2 = st.tabs(["Distribution", "Comparisons"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure(data=[go.Bar(
                y=[f'{int(r)} Stars' for r in counts.index],
                x=counts.values,
                orientation='h',
                marker_color=CHART_COLORS[:5],
                text=[f'{v:,} ({p}%)' for v, p in zip(counts.values, pcts.values)],
                textposition='outside'
            )])
            fig.update_layout(title="Rating Distribution", height=400, xaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure(data=[go.Pie(
                labels=[f'{int(r)} Stars' for r in counts.index],
                values=counts.values,
                hole=0.5,
                marker_colors=CHART_COLORS[:5]
            )])
            fig.update_layout(title="Rating Share", height=400, annotations=[
                dict(text=f'{ratings.mean():.2f}', x=0.5, y=0.5, font_size=20, showarrow=False)])
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        compare_by = st.selectbox("Compare by", ["Year", "Vote Count"])

        if compare_by == "Year":
            yearly_ratings = filtered_df.groupby(['review_year', 'rating']).size().unstack(fill_value=0)
            yearly_pct = yearly_ratings.div(yearly_ratings.sum(axis=1), axis=0) * 100
            fig = go.Figure()
            for i, rating in enumerate(yearly_pct.columns):
                fig.add_trace(go.Bar(x=yearly_pct.index, y=yearly_pct[rating], name=f'{int(rating)} Stars',
                                     marker_color=CHART_COLORS[i]))
            fig.update_layout(barmode='stack', title="Rating by Year (%)", height=450)
        else:
            voted = filtered_df[filtered_df['total_votes'] > 0].copy()
            voted['vote_cat'] = pd.cut(voted['total_votes'], bins=[0, 5, 20, 100, float('inf')],
                                       labels=['1-5', '6-20', '21-100', '100+'])
            vote_ratings = voted.groupby(['vote_cat', 'rating']).size().unstack(fill_value=0)
            vote_pct = vote_ratings.div(vote_ratings.sum(axis=1), axis=0) * 100
            fig = go.Figure()
            for i, rating in enumerate(vote_pct.columns):
                fig.add_trace(go.Bar(x=vote_pct.index.astype(str), y=vote_pct[rating], name=f'{int(rating)} Stars',
                                     marker_color=CHART_COLORS[i]))
            fig.update_layout(barmode='group', title="Rating by Vote Count (%)", height=450)

        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Detailed Breakdown")
    breakdown = pd.DataFrame({
        'Rating': [f'{int(r)} Stars' for r in counts.index],
        'Count': [f'{v:,}' for v in counts.values],
        'Percentage': [f'{p}%' for p in pcts.values],
        'Category': ['Negative', 'Negative', 'Neutral', 'Positive', 'Positive']
    })
    st.dataframe(breakdown, hide_index=True, use_container_width=True)


# ========= REVIEWER ANALYSIS =========
def render_reviewer_analysis(df, filters):
    filtered_df = get_filtered_data(df, filters)

    st.title("Reviewer Analysis")
    st.divider()

    user_stats = filtered_df.groupby('userId').agg(review_count=('productId', 'count'),
                                                   avg_rating=('rating', 'mean')).reset_index()

    def segment(c):
        if c <= 5:
            return "Casual (1-5)"
        elif c <= 50:
            return "Moderate (6-50)"
        return "Power (50+)"

    user_stats['segment'] = user_stats['review_count'].apply(segment)
    seg_order = ["Casual (1-5)", "Moderate (6-50)", "Power (50+)"]
    seg_counts = user_stats['segment'].value_counts().reindex(seg_order).fillna(0)
    seg_reviews = user_stats.groupby('segment')['review_count'].sum().reindex(seg_order).fillna(0)
    total_users = len(user_stats)
    total_reviews = seg_reviews.sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Users", format_number(total_users))
    col2.metric("Avg Reviews/User", f"{user_stats['review_count'].mean():.1f}")
    col3.metric("Max Reviews", format_number(user_stats['review_count'].max()))
    power_pct = seg_counts.get('Power (50+)', 0) / total_users * 100 if total_users > 0 else 0
    col4.metric("Power Users", f"{power_pct:.2f}%")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["Segments", "Top Reviewers", "Activity Heatmap"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure(data=[go.Bar(
                x=seg_order, y=seg_counts.values,
                marker_color=CHART_COLORS[:3],
                text=[f'{int(v):,} ({v / total_users * 100:.1f}%)' for v in seg_counts.values],
                textposition='outside'
            )])
            fig.update_layout(title="Users by Segment", height=400, yaxis_title="Users")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            user_pcts = (seg_counts / total_users * 100).values if total_users > 0 else [0, 0, 0]
            review_pcts = (seg_reviews / total_reviews * 100).values if total_reviews > 0 else [0, 0, 0]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=seg_order, y=user_pcts, name='% Users', marker_color=CHART_COLORS[0]))
            fig.add_trace(go.Bar(x=seg_order, y=review_pcts, name='% Reviews', marker_color=CHART_COLORS[1]))
            fig.update_layout(barmode='group', title="Users vs Reviews", height=400, yaxis_title="%")
            st.plotly_chart(fig, use_container_width=True)

        power_rev_pct = seg_reviews.get('Power (50+)', 0) / total_reviews * 100 if total_reviews > 0 else 0
        st.warning(
            f"**Key Insight**: Power users ({power_pct:.2f}% of users) contribute {power_rev_pct:.1f}% of reviews.")

    with tab2:
        n_top = st.slider("Top reviewers", 5, 25, 10)
        top_users = user_stats.nlargest(n_top, 'review_count')

        col1, col2 = st.columns([2, 1])

        with col1:
            fig = go.Figure(data=[go.Bar(
                y=[f'#{i + 1}' for i in range(len(top_users))],
                x=top_users['review_count'].values,
                orientation='h',
                marker_color=[CHART_COLORS[i % len(CHART_COLORS)] for i in range(len(top_users))],
                text=[f'{v:,} (Avg: {r:.1f})' for v, r in zip(top_users['review_count'], top_users['avg_rating'])],
                textposition='outside'
            )])
            fig.update_layout(title=f"Top {n_top} Reviewers", height=max(350, n_top * 30),
                              yaxis=dict(autorange='reversed'))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            top_total = top_users['review_count'].sum()
            fig = go.Figure(data=[go.Pie(
                labels=[f'Top {n_top}', 'Others'],
                values=[top_total, total_reviews - top_total],
                marker_colors=[CHART_COLORS[1], CHART_COLORS[6]],
                hole=0.4
            )])
            fig.update_layout(title="Contribution", height=300)
            st.plotly_chart(fig, use_container_width=True)
            st.metric(f"Top {n_top} Share", f"{top_total / total_reviews * 100:.2f}%" if total_reviews > 0 else "0%")

    with tab3:
        activity_bins = [0, 5, 20, 50, 100, float('inf')]
        activity_labels = ['1-5', '6-20', '21-50', '51-100', '100+']
        rating_bins = [0, 2, 3, 4, 5.1]
        rating_labels = ['Low (1-2)', 'Mid (2-3)', 'Good (3-4)', 'High (4-5)']

        user_stats['act_bin'] = pd.cut(user_stats['review_count'], bins=activity_bins, labels=activity_labels,
                                       right=False)
        user_stats['rat_bin'] = pd.cut(user_stats['avg_rating'], bins=rating_bins, labels=rating_labels, right=False)

        heatmap = user_stats.groupby(['act_bin', 'rat_bin'], observed=False).size().unstack(fill_value=0)
        heatmap = heatmap.reindex(index=activity_labels, columns=rating_labels, fill_value=0)

        fig = go.Figure(data=go.Heatmap(
            z=heatmap.values, x=rating_labels, y=activity_labels,
            colorscale='Blues', text=heatmap.values, texttemplate='%{text:,}'
        ))
        fig.update_layout(title="Activity vs Rating", height=400, xaxis_title="Avg Rating",
                          yaxis_title="Reviews Written")
        st.plotly_chart(fig, use_container_width=True)


# ========= HELPFULNESS ANALYSIS =========
def render_helpfulness_analysis(df, filters):
    filtered_df = get_filtered_data(df, filters)

    st.title("Helpfulness Analysis")
    st.divider()

    if 'helpful_yes' not in filtered_df.columns or 'total_votes' not in filtered_df.columns:
        st.warning("Helpfulness data not available.")
        return

    votes_df = filtered_df[(filtered_df['total_votes'] > 0) & (filtered_df['helpful_yes'] > 0)].copy()
    if len(votes_df) == 0:
        st.warning("No reviews with votes found.")
        return

    votes_df['helpful_ratio'] = votes_df['helpful_yes'] / votes_df['total_votes']

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Reviews with Votes", format_number(len(votes_df)))
    col2.metric("Avg Helpful Ratio", f"{votes_df['helpful_ratio'].mean():.1%}")
    col3.metric("Highly Helpful (80%+)", format_number((votes_df['helpful_ratio'] >= 0.8).sum()))
    col4.metric("Total Votes", format_number(votes_df['total_votes'].sum()))

    st.divider()

    tab1, tab2 = st.tabs(["Overview", "By Rating"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            sample = votes_df.sample(min(3000, len(votes_df)), random_state=42)
            fig = px.scatter(sample, x='total_votes', y='helpful_yes', color='rating', opacity=0.5,
                             color_continuous_scale='Viridis')
            max_v = max(sample['total_votes'].max(), sample['helpful_yes'].max())
            fig.add_trace(go.Scatter(x=[0, max_v], y=[0, max_v], mode='lines', line=dict(dash='dash', color='gray'),
                                     name='100% Line'))
            fig.update_layout(title="Helpful vs Total Votes", height=400)
            fig.update_xaxes(type='log', title='Total Votes')
            fig.update_yaxes(type='log', title='Helpful Votes')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure(data=[go.Histogram(x=votes_df['helpful_ratio'], nbinsx=40, marker_color=CHART_COLORS[2])])
            fig.update_layout(title="Helpful Ratio Distribution", height=400, xaxis_title="Ratio", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        by_rating = votes_df.groupby('rating').agg(count=('helpful_ratio', 'count'),
                                                   avg_ratio=('helpful_ratio', 'mean'),
                                                   avg_votes=('total_votes', 'mean')).reset_index()

        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure(data=[go.Bar(
                x=[f'{int(r)} Stars' for r in by_rating['rating']],
                y=by_rating['avg_ratio'],
                marker_color=CHART_COLORS[:5],
                text=[f'{v:.1%}' for v in by_rating['avg_ratio']],
                textposition='outside'
            )])
            fig.update_layout(title="Avg Helpful Ratio by Rating", height=400, yaxis_title="Ratio")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure(data=[go.Bar(
                x=[f'{int(r)} Stars' for r in by_rating['rating']],
                y=by_rating['avg_votes'],
                marker_color=CHART_COLORS[:5],
                text=[f'{v:.1f}' for v in by_rating['avg_votes']],
                textposition='outside'
            )])
            fig.update_layout(title="Avg Votes by Rating", height=400, yaxis_title="Votes")
            st.plotly_chart(fig, use_container_width=True)

        best = by_rating.loc[by_rating['avg_ratio'].idxmax()]
        st.info(
            f"**Insight**: {int(best['rating'])}-star reviews have highest helpful ratio ({best['avg_ratio']:.1%}).")


# ========= PRODUCT ANALYSIS =========
def render_product_analysis(df, filters):
    filtered_df = get_filtered_data(df, filters)

    st.title("Product Analysis")
    st.divider()

    prod_stats = filtered_df.groupby('productId').agg(
        reviews=('userId', 'count'), avg_rating=('rating', 'mean'), total_votes=('total_votes', 'sum')
    ).reset_index().sort_values('reviews', ascending=False)

    total_prods = len(prod_stats)
    total_reviews = prod_stats['reviews'].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Products", format_number(total_prods))
    col2.metric("Avg Reviews/Product", f"{prod_stats['reviews'].mean():.1f}")
    col3.metric("Median Reviews", f"{prod_stats['reviews'].median():.0f}")
    col4.metric("Max Reviews", format_number(prod_stats['reviews'].max()))

    st.divider()

    tab1, tab2, tab3 = st.tabs(["Top Products", "Distribution", "Search"])

    with tab1:
        n_prods = st.slider("Products to show", 5, 30, 10)
        top_prods = prod_stats.head(n_prods)

        col1, col2 = st.columns([2, 1])

        with col1:
            fig = go.Figure(data=[go.Bar(
                y=[f'#{i + 1}' for i in range(len(top_prods))],
                x=top_prods['reviews'].values,
                orientation='h',
                marker_color=[CHART_COLORS[i % len(CHART_COLORS)] for i in range(len(top_prods))],
                text=[f'{v:,} (Avg: {r:.1f})' for v, r in zip(top_prods['reviews'], top_prods['avg_rating'])],
                textposition='outside'
            )])
            fig.update_layout(title=f"Top {n_prods} Products", height=max(350, n_prods * 30),
                              yaxis=dict(autorange='reversed'), xaxis_title="Reviews")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            top10_share = prod_stats.head(10)['reviews'].sum() / total_reviews * 100 if total_reviews > 0 else 0
            top10_sum = prod_stats.head(10)['reviews'].sum()
            top50_sum = prod_stats.iloc[10:50]['reviews'].sum() if total_prods > 10 else 0
            others_sum = prod_stats.iloc[50:]['reviews'].sum() if total_prods > 50 else prod_stats.iloc[10:][
                'reviews'].sum()

            fig = go.Figure(data=[go.Pie(
                labels=['Top 10', 'Top 11-50', 'Others'],
                values=[top10_sum, top50_sum, others_sum],
                marker_colors=CHART_COLORS[:3], hole=0.4
            )])
            fig.update_layout(title="Concentration", height=300)
            st.plotly_chart(fig, use_container_width=True)
            st.metric("Top 10 Share", f"{top10_share:.1f}%")

        if n_prods > 1 and top_prods['reviews'].iloc[-1] > 0:
            disp = top_prods['reviews'].iloc[0] / top_prods['reviews'].iloc[-1]
            st.warning(f"**Disparity**: #1 has {disp:.1f}x more reviews than #{n_prods}")

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure(data=[go.Histogram(x=prod_stats['reviews'], nbinsx=50, marker_color=CHART_COLORS[0])])
            fig.update_layout(title="Reviews per Product", height=400)
            fig.update_xaxes(type='log', title='Reviews (log)')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure(data=[go.Histogram(x=prod_stats['avg_rating'], nbinsx=30, marker_color=CHART_COLORS[1])])
            fig.update_layout(title="Avg Rating per Product", height=400, xaxis_title="Rating")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        search_id = st.text_input("Enter Product ID", placeholder="e.g., B007FQDPL8")

        if search_id:
            prod_data = filtered_df[filtered_df['productId'].str.contains(search_id, case=False, na=False)]
            if len(prod_data) > 0:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Reviews", format_number(len(prod_data)))
                col2.metric("Avg Rating", f"{prod_data['rating'].mean():.2f}")
                col3.metric("Users", format_number(prod_data['userId'].nunique()))
                col4.metric("Total Votes", format_number(prod_data['total_votes'].sum()))

                rating_ct = prod_data['rating'].value_counts().sort_index()
                fig = go.Figure(data=[go.Bar(x=rating_ct.index.astype(int), y=rating_ct.values,
                                             marker_color=CHART_COLORS[:len(rating_ct)])])
                fig.update_layout(title=f"Ratings for {search_id}", height=300, xaxis_title="Rating",
                                  yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(prod_data[['userId', 'rating', 'helpful_yes', 'total_votes', 'review_date']].head(20),
                             use_container_width=True)
            else:
                st.warning("No products found.")

        st.divider()
        st.subheader("Top Products Table")
        disp_df = prod_stats.head(50).copy()
        disp_df.insert(0, 'Rank', range(1, len(disp_df) + 1))
        disp_df.columns = ['Rank', 'Product ID', 'Reviews', 'Avg Rating', 'Total Votes']
        st.dataframe(disp_df.style.format({'Avg Rating': '{:.2f}', 'Reviews': '{:,}', 'Total Votes': '{:,}'}),
                     use_container_width=True, hide_index=True)


# ========= MAIN =========
def main():
    st.title("Amazon Movie Reviews EDA")

    # Data source selection
    data_source = st.radio(
        "Select Data Source",
        ["Use Sample Data (Demo)", "Upload Your Data", "Load from Local Path"],
        horizontal=True
    )

    df = None

    if data_source == "Use Sample Data (Demo)":
        with st.spinner("Loading sample data..."):
            df = load_sample_data()
        st.success("Sample data loaded (10,000 reviews for demonstration)")

    elif data_source == "Upload Your Data":
        uploaded_file = st.file_uploader("Upload CSV or Parquet file", type=['csv', 'parquet'])
        if uploaded_file:
            with st.spinner("Loading uploaded data..."):
                df = load_data_from_file(uploaded_file)
            if df is not None:
                st.success(f"Loaded {len(df):,} rows")

    else:  # Local path
        local_path = st.text_input(
            "Enter local data folder path",
            value="/Users/eyinadeiyanuoluwa/Desktop/ALY6110 final project/data_processed/reviews_parquet_stage6_clean"
        )
        if st.button("Load Data"):
            with st.spinner("Loading local data..."):
                df = load_local_data(local_path)
            if df is not None:
                st.success(f"Loaded {len(df):,} rows")
            else:
                st.error("Could not load data from the specified path")

    if df is None:
        st.info("Please select a data source to begin.")
        return

    st.divider()

    section, filters = render_sidebar(df)

    if section == "Overview":
        render_overview(df, filters)
    elif section == "Time Trends":
        render_time_trends(df, filters)
    elif section == "Rating Analysis":
        render_rating_analysis(df, filters)
    elif section == "Reviewer Analysis":
        render_reviewer_analysis(df, filters)
    elif section == "Helpfulness Analysis":
        render_helpfulness_analysis(df, filters)
    elif section == "Product Analysis":
        render_product_analysis(df, filters)


if __name__ == "__main__":
    main()