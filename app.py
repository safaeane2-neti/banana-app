# --- NEW Clean & Minimalist HTML Dashboard ---
def create_dashboard_html(temp, humidity, ethylene, day, shelf_life, economics_info, optimal_info):
    
    # --- CUSTOMIZE YOUR MAIN COLOR HERE ---
    # Try these: '#3498db' (Blue), '#2ecc71' (Green), '#9b59b6' (Purple), '#e74c3c' (Red)
    main_color = '#f39c12' # A nice, soft orange

    ripeness = economics.calculate_ripeness(shelf_life, day)
    economics = BananaEconomics()
    market, _, emoji = economics.get_market_segment(ripeness)

    html = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; max-width: 900px; margin: 40px auto; color: #333;">
        
        <!-- Header -->
        <div style="text-align: center; margin-bottom: 40px;">
            <h1 style="font-size: 42px; font-weight: 700; color: #333; margin: 0;">
                Twinsie Dashboard
            </h1>
            <p style="font-size: 18px; color: #777; margin-top: 8px;">
                AI-Powered Ripeness & Profitability Analysis
            </p>
        </div>
        
        <!-- Key Metrics -->
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 30px;">
            <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; border-left: 4px solid {main_color};">
                <div style="font-size: 24px; margin-bottom: 5px;">{emoji}</div>
                <div style="font-size: 14px; color: #777; text-transform: uppercase; letter-spacing: 0.5px;">Market</div>
                <div style="font-size: 20px; font-weight: 600; color: #333;">{market}</div>
            </div>
            <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center;">
                <div style="font-size: 24px; margin-bottom: 5px;">⏳</div>
                <div style="font-size: 14px; color: #777; text-transform: uppercase; letter-spacing: 0.5px;">Shelf Life</div>
                <div style="font-size: 20px; font-weight: 600; color: #333;">{shelf_life:.1f} days</div>
            </div>
            <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center;">
                <div style="font-size: 24px; margin-bottom: 5px;">💵</div>
                <div style="font-size: 14px; color: #777; text-transform: uppercase; letter-spacing: 0.5px;">Net Value</div>
                <div style="font-size: 20px; font-weight: 600; color: #333;">${economics_info['net_value']:.2f}/kg</div>
            </div>
            <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center;">
                <div style="font-size: 24px; margin-bottom: 5px;">🎯</div>
                <div style="font-size: 14px; color: #777; text-transform: uppercase; letter-spacing: 0.5px;">Optimal Day</div>
                <div style="font-size: 20px; font-weight: 600; color: #333;">Day {optimal_info['day']}</div>
            </div>
        </div>

        <!-- Details Section -->
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            
            <!-- Sensor Data -->
            <div style="background: #ffffff; padding: 25px; border-radius: 8px; border: 1px solid #e9ecef;">
                <h3 style="font-size: 18px; font-weight: 600; margin-top: 0; margin-bottom: 20px; color: #333;">Sensor Readings</h3>
                <div style="display: flex; justify-content: space-between; margin-bottom: 12px;"><span style="color: #777;">Temperature</span><span style="font-weight: 500;">{temp:.1f}°C</span></div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 12px;"><span style="color: #777;">Humidity</span><span style="font-weight: 500;">{humidity:.0f}%</span></div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 12px;"><span style="color: #777;">Ethylene</span><span style="font-weight: 500;">{ethylene:.1f} ppm</span></div>
                <div style="display: flex; justify-content: space-between;"><span style="color: #777;">Current Day</span><span style="font-weight: 500;">{day:.1f}</span></div>
            </div>
            
            <!-- Projection -->
            <div style="background: #ffffff; padding: 25px; border-radius: 8px; border: 1px solid #e9ecef;">
                <h3 style="font-size: 18px; font-weight: 600; margin-top: 0; margin-bottom: 20px; color: #333;">Container Projection</h3>
                <div style="text-align: center;">
                    <div style="font-size: 32px; font-weight: 700; color: {main_color};">
                        ${economics_info['net_value'] * 18000:,.0f}
                    </div>
                    <div style="font-size: 14px; color: #777; margin-top: 5px;">Current Value (18,000 kg)</div>
                </div>
                <hr style="border: none; border-top: 1px solid #e9ecef; margin: 20px 0;">
                <div style="text-align: center;">
                    <div style="font-size: 24px; font-weight: 600; color: #2ecc71;">
                        +${(optimal_info['value'] - economics_info['net_value']) * 18000:,.0f}
                    </div>
                    <div style="font-size: 14px; color: #777; margin-top: 5px;">Potential Gain with Optimization</div>
                </div>
            </div>
        </div>
        
        <!-- Recommendation -->
        <div style="background: {main_color}; color: white; padding: 20px; border-radius: 8px; text-align: center; margin-top: 20px;">
            <p style="font-size: 18px; font-weight: 600; margin: 0;">
                {"Ship now for optimal profit." if abs(day - optimal_info['day']) < 0.5 
                 else f"Wait {optimal_info['day'] - day:.0f} days for optimal profit." if day < optimal_info['day']
                 else "Optimal shipping window has passed."}
            </p>
        </div>

    </div>
    """
    
    return st.components.v1.html(html, height=700)
       
