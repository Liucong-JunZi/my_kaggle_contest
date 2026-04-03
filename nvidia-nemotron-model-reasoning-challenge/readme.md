⏺ # 全量跑，40 workers，2个key轮询                                                                                      
  nohup ./generate_cot --env .env2 --workers 40 > cot_output/run.log 2>&1 &
                                                                                                                        
  # 查看进度                                                                                                            
  tail -f cot_output/run.log                                                                                            
                                                                                                                        
  # 看checkpoint统计                                                                                                    
  cat cot_output/checkpoint.json | python3 -c "import json,sys; c=json.load(sys.stdin); print(f'Solved:                 
  {len(c[\"solved\"])}, Distilled: {len(c[\"distilled\"])}, Techniques: {len(c[\"techniques_extracted\"])}')" 