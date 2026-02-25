if st.button("🚀 Sinh AI phân tích và tư vấn"):
        if imputed_df.empty:
            st.info("Chưa có dữ liệu để phân tích.")
        else:
            prompt = build_ai_prompt(
                audience=audience,
                country_label=sel_country_name,
                year_range=f"{start_year}-{end_year}",
                stats_df=stats_df,
                corr_df=corr_df,
                selected_cols=[c for c in imputed_df.columns if c != "Year"]
            )
            if not GEMINI_OK:
                st.warning("⚠️ Mô-đun AI chưa sẵn sàng (thiếu thư viện google-generativeai). Bạn vui lòng cập nhật file requirements.txt.")
            else:
                # Ưu tiên lấy từ st.secrets (khi deploy), nếu không có thì lấy từ biến môi trường
                api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", "")).strip()
                
                if not api_key:
                    st.warning("⚠️ Chưa phát hiện GEMINI_API_KEY. Vui lòng đặt trong st.secrets hoặc cấu hình biến môi trường.")
                else:
                    try:
                        genai.configure(api_key=api_key)
                        
                        # Khởi tạo mô hình bản lõi (tương thích 100% mọi thư viện)
                        model = genai.GenerativeModel("gemini-pro")
                        
                        # Cấu hình các tham số sinh văn bản
                        generation_config = genai.types.GenerationConfig(
                            temperature=0.4,
                            max_output_tokens=900,
                        )
                        
                        # Gộp lệnh system vào trực tiếp prompt để tránh lỗi phiên bản cũ
                        final_prompt = "Bạn là chuyên gia kinh tế vĩ mô & tài chính, viết ngắn gọn, súc tích, dùng tiêu đề tiếng Việt.\n\n" + prompt
                        
                        # Gọi API sinh nội dung
                        response = model.generate_content(final_prompt, generation_config=generation_config)
                        
                        st.markdown(response.text)
                    except Exception as e:
                        st.error(f"Lỗi khi gọi Gemini: {e}")

# Footer
st.caption("© 2025 — Viet Macro Intelligence • Nguồn: " + "; ".join(source_list))
