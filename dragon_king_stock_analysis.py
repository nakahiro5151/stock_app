"""
ソネット氏のドラゴンキング理論に基づく株価暴落予兆分析システム
Google Colab用

ドラゴンキング理論の主要指標:
1. ログ周期的振動 (Log-Periodic Oscillations)
2. 加速度的価格上昇
3. ボラティリティの異常増加
4. 相関構造の変化
5. 臨界点への収束パターン
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import optimize, signal
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Google Colab環境設定
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
except:
    IN_COLAB = False

# 日本語フォント設定（Google Colab対応）
import matplotlib.font_manager as fm

# Google Colabで日本語フォントを設定
try:
    # Noto Sans CJK JPフォントを試行
    plt.rcParams['font.family'] = 'DejaVu Sans'
    # 日本語フォントの設定を試行
    font_path = '/usr/share/fonts/truetype/fonts-japanese-gothic.ttf'
    if fm.fontManager.findfont(fm.FontProperties(fname=font_path)) != fm.get_default_font():
        plt.rcParams['font.family'] = 'fonts-japanese-gothic'
    else:
        # フォールバック設定
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
except:
    # フォント設定に失敗した場合のフォールバック
    plt.rcParams['font.family'] = 'DejaVu Sans'

plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化け防止

class DragonKingAnalyzer:
    """ドラゴンキング理論に基づく株価分析クラス"""
    
    def __init__(self, symbol, period="3y"):
        """
        初期化
        
        Args:
            symbol (str): 株式シンボル (例: "^N225", "AAPL", "7203.T")
            period (str): データ取得期間 ("1y", "2y", "5y", "max")
        """
        self.symbol = symbol
        self.period = period
        self.data = None
        self.analysis_results = {}
        
    def fetch_data(self):
        """株価データを取得"""
        print(f"📈 {self.symbol} の株価データを取得中...")
        
        try:
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=self.period)
            
            if self.data.empty:
                raise ValueError(f"データが取得できませんでした: {self.symbol}")
                
            print(f"✅ データ取得完了: {len(self.data)} 日分")
            return True
            
        except Exception as e:
            print(f"❌ データ取得エラー: {e}")
            return False
    
    def calculate_log_periodic_oscillations(self, window=60):
        """
        ログ周期的振動の検出
        
        Args:
            window (int): 分析ウィンドウサイズ
        """
        print("🔍 ログ周期的振動を分析中...")
        
        prices = self.data['Close'].values
        log_prices = np.log(prices)
        
        # 移動平均からの乖離を計算
        rolling_mean = pd.Series(log_prices).rolling(window=window).mean()
        deviations = log_prices - rolling_mean
        
        # フーリエ変換で周期性を検出
        fft = np.fft.fft(deviations[window:])
        frequencies = np.fft.fftfreq(len(fft))
        power_spectrum = np.abs(fft) ** 2
        
        # 主要な周期を特定
        dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
        dominant_period = 1 / abs(frequencies[dominant_freq_idx]) if frequencies[dominant_freq_idx] != 0 else 0
        
        # 振動の強度を計算
        oscillation_strength = np.std(deviations[window:])
        
        self.analysis_results['log_periodic'] = {
            'deviations': deviations,
            'dominant_period': dominant_period,
            'oscillation_strength': oscillation_strength,
            'power_spectrum': power_spectrum,
            'frequencies': frequencies
        }
        
        print(f"📊 主要周期: {dominant_period:.1f} 日")
        print(f"📊 振動強度: {oscillation_strength:.4f}")
    
    def calculate_acceleration_pattern(self, window=30):
        """
        価格上昇の加速度パターンを分析
        
        Args:
            window (int): 移動平均ウィンドウ
        """
        print("🚀 価格加速度パターンを分析中...")
        
        prices = self.data['Close'].values
        
        # 1次微分（速度）
        velocity = np.gradient(prices)
        
        # 2次微分（加速度）
        acceleration = np.gradient(velocity)
        
        # 移動平均で平滑化
        smooth_velocity = pd.Series(velocity).rolling(window=window).mean()
        smooth_acceleration = pd.Series(acceleration).rolling(window=window).mean()
        
        # 加速度の異常値を検出
        acc_threshold = np.percentile(np.abs(smooth_acceleration.dropna()), 95)
        acceleration_anomalies = np.abs(smooth_acceleration) > acc_threshold
        
        self.analysis_results['acceleration'] = {
            'velocity': velocity,
            'acceleration': acceleration,
            'smooth_velocity': smooth_velocity,
            'smooth_acceleration': smooth_acceleration,
            'anomalies': acceleration_anomalies,
            'threshold': acc_threshold
        }
        
        anomaly_count = acceleration_anomalies.sum()
        print(f"📊 加速度異常検出: {anomaly_count} 回")
    
    def calculate_volatility_regime(self, window=20):
        """
        ボラティリティレジームの変化を分析
        
        Args:
            window (int): ボラティリティ計算ウィンドウ
        """
        print("📈 ボラティリティレジームを分析中...")
        
        returns = self.data['Close'].pct_change().dropna()
        
        # 実現ボラティリティ（移動標準偏差）
        realized_vol = returns.rolling(window=window).std() * np.sqrt(252)
        
        # ボラティリティの変化率
        vol_change = realized_vol.pct_change()
        
        # レジーム変化の検出（ボラティリティが急激に変化）
        vol_threshold = np.percentile(np.abs(vol_change.dropna()), 90)
        regime_changes = np.abs(vol_change) > vol_threshold
        
        # GARCH効果の検出（ボラティリティクラスタリング）
        vol_autocorr = realized_vol.autocorr(lag=1)
        
        self.analysis_results['volatility'] = {
            'returns': returns,
            'realized_vol': realized_vol,
            'vol_change': vol_change,
            'regime_changes': regime_changes,
            'vol_autocorr': vol_autocorr,
            'threshold': vol_threshold
        }
        
        regime_count = regime_changes.sum()
        print(f"📊 ボラティリティレジーム変化: {regime_count} 回")
        print(f"📊 ボラティリティ自己相関: {vol_autocorr:.3f}")
    
    def calculate_critical_point_indicators(self):
        """
        臨界点指標の計算
        """
        print("⚠️ 臨界点指標を計算中...")
        
        prices = self.data['Close'].values
        log_prices = np.log(prices)
        
        # 価格の累積変化率
        cumulative_returns = np.cumsum(np.diff(log_prices))
        
        # トレンドの強度（線形回帰の傾き）
        x = np.arange(len(log_prices))
        slope, intercept = np.polyfit(x, log_prices, 1)
        
        # R²値（トレンドの説明力）
        y_pred = slope * x + intercept
        ss_res = np.sum((log_prices - y_pred) ** 2)
        ss_tot = np.sum((log_prices - np.mean(log_prices)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # 価格の分散の時系列変化
        rolling_var = pd.Series(log_prices).rolling(window=30).var()
        
        self.analysis_results['critical_point'] = {
            'cumulative_returns': cumulative_returns,
            'trend_slope': slope,
            'r_squared': r_squared,
            'rolling_variance': rolling_var
        }
        
        print(f"📊 トレンド傾き: {slope:.6f}")
        print(f"📊 トレンド説明力 (R²): {r_squared:.3f}")
    
    def calculate_dragon_king_score(self):
        """
        ドラゴンキング総合スコアの計算
        """
        print("🐉 ドラゴンキングスコアを計算中...")
        
        # 各指標を正規化してスコア化
        scores = {}
        
        # 1. ログ周期的振動スコア（振動が強いほど高スコア）
        osc_strength = self.analysis_results['log_periodic']['oscillation_strength']
        scores['oscillation'] = min(osc_strength * 10, 1.0)
        
        # 2. 加速度スコア（異常な加速度が多いほど高スコア）
        acc_anomalies = self.analysis_results['acceleration']['anomalies'].sum()
        scores['acceleration'] = min(acc_anomalies / 10, 1.0)
        
        # 3. ボラティリティスコア（レジーム変化が多いほど高スコア）
        vol_changes = self.analysis_results['volatility']['regime_changes'].sum()
        scores['volatility'] = min(vol_changes / 5, 1.0)
        
        # 4. トレンドスコア（強いトレンドほど高スコア）
        r_squared = self.analysis_results['critical_point']['r_squared']
        scores['trend'] = r_squared
        
        # 総合スコア（重み付き平均）
        weights = {'oscillation': 0.3, 'acceleration': 0.25, 'volatility': 0.25, 'trend': 0.2}
        total_score = sum(scores[key] * weights[key] for key in scores)
        
        self.analysis_results['dragon_king_score'] = {
            'individual_scores': scores,
            'total_score': total_score,
            'risk_level': self._get_risk_level(total_score)
        }
        
        print(f"🐉 ドラゴンキング総合スコア: {total_score:.3f}")
        print(f"⚠️ リスクレベル: {self._get_risk_level(total_score)}")
        
        return total_score
    
    def _get_risk_level(self, score):
        """リスクレベルの判定"""
        if score >= 0.7:
            return "🔴 極めて高い"
        elif score >= 0.5:
            return "🟠 高い"
        elif score >= 0.3:
            return "🟡 中程度"
        else:
            return "🟢 低い"
    
    def plot_comprehensive_analysis(self):
        """包括的な分析結果をプロット"""
        import matplotlib.dates as mdates
        from matplotlib.dates import DateFormatter
        
        # 日本語フォント設定を再確認
        try:
            import matplotlib.font_manager as fm
            # 利用可能な日本語フォントを検索
            japanese_fonts = [f.name for f in fm.fontManager.ttflist if 'japan' in f.name.lower() or 'noto' in f.name.lower() or 'cjk' in f.name.lower()]
            if japanese_fonts:
                plt.rcParams['font.family'] = japanese_fonts[0]
        except:
            pass
        
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        fig.suptitle(f'{self.symbol} - Dragon King Theory Analysis', fontsize=16, fontweight='bold')
        
        dates = self.data.index
        
        # 1. 価格チャートとログ周期的振動
        ax1 = axes[0, 0]
        ax1.plot(dates, self.data['Close'], 'b-', linewidth=2, label='Stock Price')
        ax1.set_title('Stock Price and Log-Periodic Oscillations (Log Scale)')
        ax1.set_ylabel('Price (Log Scale)')
        ax1.set_yscale('log')  # ログスケール設定
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # X軸の日付フォーマット設定（3年間対応）
        ax1.xaxis.set_major_formatter(DateFormatter('%m/%d\n%y'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # 3ヶ月間隔
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha='center')
        
        # 振動の可視化
        ax1_twin = ax1.twinx()
        deviations = self.analysis_results['log_periodic']['deviations']
        if len(deviations) == len(dates):
            ax1_twin.plot(dates, deviations, 'r-', alpha=0.7, label='Log-Periodic Oscillation')
            ax1_twin.set_ylabel('Oscillation Strength', color='r')
            ax1_twin.legend(loc='upper right')
        
        # 2. 加速度パターン
        ax2 = axes[0, 1]
        acceleration = self.analysis_results['acceleration']['smooth_acceleration']
        if len(acceleration) == len(dates):
            ax2.plot(dates, acceleration, 'g-', linewidth=2)
            ax2.axhline(y=self.analysis_results['acceleration']['threshold'], 
                       color='r', linestyle='--', label='Anomaly Threshold')
            ax2.axhline(y=-self.analysis_results['acceleration']['threshold'], 
                       color='r', linestyle='--')
        ax2.set_title('Price Acceleration Pattern')
        ax2.set_ylabel('Acceleration')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # X軸の日付フォーマット設定（3年間対応）
        ax2.xaxis.set_major_formatter(DateFormatter('%m/%d\n%y'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # 3ヶ月間隔
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha='center')
        
        # 3. ボラティリティレジーム
        ax3 = axes[1, 0]
        vol_data = self.analysis_results['volatility']['realized_vol']
        ax3.plot(vol_data.index, vol_data.values, 'purple', linewidth=2)
        ax3.set_title('Realized Volatility')
        ax3.set_ylabel('Annualized Volatility')
        ax3.grid(True, alpha=0.3)
        
        # X軸の日付フォーマット設定（3年間対応）
        ax3.xaxis.set_major_formatter(DateFormatter('%m/%d\n%y'))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # 3ヶ月間隔
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=0, ha='center')
        
        # 4. リターン分布
        ax4 = axes[1, 1]
        returns = self.analysis_results['volatility']['returns']
        ax4.hist(returns, bins=50, alpha=0.7, color='orange', edgecolor='black')
        ax4.axvline(x=returns.mean(), color='r', linestyle='--', label=f'Mean: {returns.mean():.4f}')
        ax4.set_title('Daily Return Distribution')
        ax4.set_xlabel('Return')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 累積リターン
        ax5 = axes[2, 0]
        cumulative_returns = self.analysis_results['critical_point']['cumulative_returns']
        if len(cumulative_returns) == len(dates) - 1:
            ax5.plot(dates[1:], cumulative_returns, 'navy', linewidth=2)
        ax5.set_title('Cumulative Returns')
        ax5.set_ylabel('Cumulative Return')
        ax5.grid(True, alpha=0.3)
        
        # X軸の日付フォーマット設定（3年間対応）
        ax5.xaxis.set_major_formatter(DateFormatter('%m/%d\n%y'))
        ax5.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # 3ヶ月間隔
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=0, ha='center')
        
        # 6. ドラゴンキングスコア
        ax6 = axes[2, 1]
        scores = self.analysis_results['dragon_king_score']['individual_scores']
        score_names = ['Oscillation', 'Acceleration', 'Volatility', 'Trend']
        score_values = [scores['oscillation'], scores['acceleration'], scores['volatility'], scores['trend']]
        
        bars = ax6.bar(score_names, score_values, 
                      color=['red', 'orange', 'yellow', 'green'])
        ax6.set_title('Dragon King Indicator Scores')
        ax6.set_ylabel('Score')
        ax6.set_ylim(0, 1)
        
        # スコア値をバーの上に表示
        for bar, value in zip(bars, score_values):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 総合スコアを大きく表示
        total_score = self.analysis_results['dragon_king_score']['total_score']
        risk_level = self.analysis_results['dragon_king_score']['risk_level']
        
        print("\n" + "="*60)
        print(f"Dragon King Comprehensive Analysis Results")
        print("="*60)
        print(f"Symbol: {self.symbol}")
        print(f"Total Score: {total_score:.3f}")
        print(f"Risk Level: {risk_level}")
        print("="*60)
    
    def run_full_analysis(self):
        """完全な分析を実行"""
        print(f"🐉 {self.symbol} のドラゴンキング理論分析を開始")
        print("="*60)
        
        # データ取得
        if not self.fetch_data():
            return False
        
        # 各分析を実行
        self.calculate_log_periodic_oscillations()
        self.calculate_acceleration_pattern()
        self.calculate_volatility_regime()
        self.calculate_critical_point_indicators()
        self.calculate_dragon_king_score()
        
        # 結果をプロット
        self.plot_comprehensive_analysis()
        
        return True

# 使用例とデモ
def demo_analysis():
    """デモ分析の実行"""
    print("🚀 ドラゴンキング理論株価分析デモ")
    print("="*60)
    
    # 分析対象銘柄
    symbols = [
        "^N225",    # 日経平均
        "^GSPC",    # S&P500
        "AAPL",     # Apple
        "7203.T",   # トヨタ
    ]
    
    results = {}
    
    for symbol in symbols:
        print(f"\n📊 {symbol} を分析中...")
        analyzer = DragonKingAnalyzer(symbol, period="1y")
        
        if analyzer.run_full_analysis():
            results[symbol] = analyzer.analysis_results
        else:
            print(f"❌ {symbol} の分析に失敗しました")
    
    return results

if __name__ == "__main__":
    # 必要なライブラリのインストール（Colab用）
    if IN_COLAB:
        print("📦 必要なライブラリをインストール中...")
        import subprocess
        import sys
        
        packages = ["yfinance"]
        for package in packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # デモ実行
    demo_results = demo_analysis()