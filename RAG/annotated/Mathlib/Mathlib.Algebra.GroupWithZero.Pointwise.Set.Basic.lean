                                                    /-
                                                      α : Type u_1
                                                      inst✝ : MulZeroClass α
                                                      s : Set α
                                                      ⊢ HasSubset.Subset (HMul.hMul s 0) 0
                                                    -/
lemma mul_zero_subset (s : Set α) : s * 0 ⊆ 0 := by simp [subset_def, mem_mul]
                                                    /-
                                                      🎉 no goals
                                                    -/

                                                    /-
                                                      α : Type u_1
                                                      inst✝ : MulZeroClass α
                                                      s : Set α
                                                      ⊢ HasSubset.Subset (HMul.hMul 0 s) 0
                                                    -/
lemma zero_mul_subset (s : Set α) : 0 * s ⊆ 0 := by simp [subset_def, mem_mul]
                                                    /-
                                                      🎉 no goals
                                                    -/


lemma Nonempty.mul_zero (hs : s.Nonempty) : s * 0 = 0 :=
                                   /-
                                     α : Type u_1
                                     inst✝ : MulZeroClass α
                                     s : Set α
                                     hs : s.Nonempty
                                     ⊢ HasSubset.Subset 0 (HMul.hMul s 0)
                                   -/
  s.mul_zero_subset.antisymm <| by simpa [mem_mul] using hs
                                   /-
                                     🎉 no goals
                                   -/


lemma Nonempty.zero_mul (hs : s.Nonempty) : 0 * s = 0 :=
                                   /-
                                     α : Type u_1
                                     inst✝ : MulZeroClass α
                                     s : Set α
                                     hs : s.Nonempty
                                     ⊢ HasSubset.Subset 0 (HMul.hMul 0 s)
                                   -/
  s.zero_mul_subset.antisymm <| by simpa [mem_mul] using hs
                                   /-
                                     🎉 no goals
                                   -/


                                                    /-
                                                      α : Type u_1
                                                      inst✝ : GroupWithZero α
                                                      s : Set α
                                                      ⊢ HasSubset.Subset (HDiv.hDiv s 0) 0
                                                    -/
lemma div_zero_subset (s : Set α) : s / 0 ⊆ 0 := by simp [subset_def, mem_div]
                                                    /-
                                                      🎉 no goals
                                                    -/

                                                    /-
                                                      α : Type u_1
                                                      inst✝ : GroupWithZero α
                                                      s : Set α
                                                      ⊢ HasSubset.Subset (HDiv.hDiv 0 s) 0
                                                    -/
lemma zero_div_subset (s : Set α) : 0 / s ⊆ 0 := by simp [subset_def, mem_div]
                                                    /-
                                                      🎉 no goals
                                                    -/


lemma Nonempty.div_zero (hs : s.Nonempty) : s / 0 = 0 :=
                                   /-
                                     α : Type u_1
                                     inst✝ : GroupWithZero α
                                     s : Set α
                                     hs : s.Nonempty
                                     ⊢ HasSubset.Subset 0 (HDiv.hDiv s 0)
                                   -/
  s.div_zero_subset.antisymm <| by simpa [mem_div] using hs
                                   /-
                                     🎉 no goals
                                   -/


lemma Nonempty.zero_div (hs : s.Nonempty) : 0 / s = 0 :=
                                   /-
                                     α : Type u_1
                                     inst✝ : GroupWithZero α
                                     s : Set α
                                     hs : s.Nonempty
                                     ⊢ HasSubset.Subset 0 (HDiv.hDiv 0 s)
                                   -/
  s.zero_div_subset.antisymm <| by simpa [mem_div] using hs
                                   /-
                                     🎉 no goals
                                   -/


