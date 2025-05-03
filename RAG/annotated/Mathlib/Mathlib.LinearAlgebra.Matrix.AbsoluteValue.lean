theorem det_le {A : Matrix n n R} {abv : AbsoluteValue R S} {x : S} (hx : ∀ i j, abv (A i j) ≤ x) :
    abv A.det ≤ Nat.factorial (Fintype.card n) • x ^ Fintype.card n :=
  calc
    abv A.det = abv (∑ σ : Perm n, Perm.sign σ • ∏ i, A (σ i) i) := congr_arg abv (det_apply _)
    _ ≤ ∑ σ : Perm n, abv (Perm.sign σ • ∏ i, A (σ i) i) := abv.sum_le _ _
    _ = ∑ σ : Perm n, ∏ i, abv (A (σ i) i) :=
                                   /-
                                     R : Type u_1
                                     S : Type u_2
                                     inst✝⁴ : CommRing R
                                     inst✝³ : Nontrivial R
                                     inst✝² : LinearOrderedCommRing S
                                     n : Type u_3
                                     inst✝¹ : Fintype n
                                     inst✝ : DecidableEq n
                                     A : Matrix n n R
                                     abv : AbsoluteValue R S
                                     x : S
                                     hx : ∀ (i j : n), LE.le (abv (A i j)) x
                                     σ : Equiv.Perm n
                                     x✝ : Membership.mem Finset.univ σ
                                     ⊢ Eq (abv (HSMul.hSMul (Equiv.Perm.sign σ) (Finset.univ.prod fun i => A (σ i)  …
                                   -/
      (sum_congr rfl fun σ _ => by rw [abv.map_units_int_smul, abv.map_prod])
                                   /-
                                     🎉 no goals
                                   -/
    _ ≤ ∑ _σ : Perm n, ∏ _i : n, x :=
      (sum_le_sum fun _ _ => prod_le_prod (fun _ _ => abv.nonneg _) fun _ _ => hx _ _)
    _ = ∑ _σ : Perm n, x ^ Fintype.card n :=
                                   /-
                                     R : Type u_1
                                     S : Type u_2
                                     inst✝⁴ : CommRing R
                                     inst✝³ : Nontrivial R
                                     inst✝² : LinearOrderedCommRing S
                                     n : Type u_3
                                     inst✝¹ : Fintype n
                                     inst✝ : DecidableEq n
                                     A : Matrix n n R
                                     abv : AbsoluteValue R S
                                     x : S
                                     hx : ∀ (i j : n), LE.le (abv (A i j)) x
                                     x✝¹ : Equiv.Perm n
                                     x✝ : Membership.mem Finset.univ x✝¹
                                     ⊢ Eq (Finset.univ.prod fun _i => x) (HPow.hPow x (Fintype.card n))
                                   -/
      (sum_congr rfl fun _ _ => by rw [prod_const, Finset.card_univ])
                                   /-
                                     🎉 no goals
                                   -/
    _ = Nat.factorial (Fintype.card n) • x ^ Fintype.card n := by
      /-
        R : Type u_1
        S : Type u_2
        inst✝⁴ : CommRing R
        inst✝³ : Nontrivial R
        inst✝² : LinearOrderedCommRing S
        n : Type u_3
        inst✝¹ : Fintype n
        inst✝ : DecidableEq n
        A : Matrix n n R
        abv : AbsoluteValue R S
        x : S
        hx : ∀ (i j : n), LE.le (abv (A i j)) x
        ⊢ Eq (Finset.univ.sum fun _σ => HPow.hPow x (Fintype.card n)) (HSMul.hSMul (Fi …
      -/
      rw [sum_const, Finset.card_univ, Fintype.card_perm]
      /-
        🎉 no goals
      -/


theorem det_sum_le {ι : Type*} (s : Finset ι) {A : ι → Matrix n n R} {abv : AbsoluteValue R S}
    {x : S} (hx : ∀ k i j, abv (A k i j) ≤ x) :
    abv (det (∑ k ∈ s, A k)) ≤
      Nat.factorial (Fintype.card n) • (#s• x) ^ Fintype.card n :=
  det_le fun i j =>
    calc
                                                              /-
                                                                R : Type u_1
                                                                S : Type u_2
                                                                inst✝⁴ : CommRing R
                                                                inst✝³ : Nontrivial R
                                                                inst✝² : LinearOrderedCommRing S
                                                                n : Type u_3
                                                                inst✝¹ : Fintype n
                                                                inst✝ : DecidableEq n
                                                                ι : Type u_4
                                                                s : Finset ι
                                                                A : ι → Matrix n n R
                                                                abv : AbsoluteValue R S
                                                                x : S
                                                                hx : ∀ (k : ι) (i j : n), LE.le (abv (A k i j)) x
                                                                i j : n
                                                                ⊢ Eq (abv (s.sum (fun k => A k) i j)) (abv (s.sum fun k => A k i j))
                                                              -/
      abv ((∑ k ∈ s, A k) i j) = abv (∑ k ∈ s, A k i j) := by simp only [sum_apply]
                                                              /-
                                                                🎉 no goals
                                                              -/
      _ ≤ ∑ k ∈ s, abv (A k i j) := abv.sum_le _ _
      _ ≤ ∑ _k ∈ s, x := sum_le_sum fun k _ => hx k i j
      _ = #s • x := sum_const _


theorem det_sum_smul_le {ι : Type*} (s : Finset ι) {c : ι → R} {A : ι → Matrix n n R}
    {abv : AbsoluteValue R S} {x : S} (hx : ∀ k i j, abv (A k i j) ≤ x) {y : S}
    (hy : ∀ k, abv (c k) ≤ y) :
    abv (det (∑ k ∈ s, c k • A k)) ≤
      Nat.factorial (Fintype.card n) • (#s • y * x) ^ Fintype.card n := by
  simpa only [smul_mul_assoc] using
    det_sum_le s fun k i j =>
      calc
        abv (c k * A k i j) = abv (c k) * abv (A k i j) := abv.map_mul _ _
        _ ≤ y * x := mul_le_mul (hy k) (hx k i j) (abv.nonneg _) ((abv.nonneg _).trans (hy k))


