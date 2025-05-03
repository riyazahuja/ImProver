@[simp]
theorem add_mem_centralizer [Distrib M] (ha : a ∈ centralizer S) (hb : b ∈ centralizer S) :
                                            /-
                                              M : Type u_1
                                              S : Set M
                                              a b : M
                                              inst✝ : Distrib M
                                              ha : Membership.mem S.centralizer a
                                              hb : Membership.mem S.centralizer b
                                              c : M
                                              hc : Membership.mem S c
                                              ⊢ Eq (HMul.hMul c (HAdd.hAdd a b)) (HMul.hMul (HAdd.hAdd a b) c)
                                            -/
    a + b ∈ centralizer S := fun c hc => by rw [add_mul, mul_add, ha c hc, hb c hc]
                                            /-
                                              🎉 no goals
                                            -/


@[simp]
theorem neg_mem_centralizer [Mul M] [HasDistribNeg M] (ha : a ∈ centralizer S) :
                                         /-
                                           M : Type u_1
                                           S : Set M
                                           a : M
                                           inst✝¹ : Mul M
                                           inst✝ : HasDistribNeg M
                                           ha : Membership.mem S.centralizer a
                                           c : M
                                           hc : Membership.mem S c
                                           ⊢ Eq (HMul.hMul c (Neg.neg a)) (HMul.hMul (Neg.neg a) c)
                                         -/
    -a ∈ centralizer S := fun c hc => by rw [mul_neg, ha c hc, neg_mul]
                                         /-
                                           🎉 no goals
                                         -/


