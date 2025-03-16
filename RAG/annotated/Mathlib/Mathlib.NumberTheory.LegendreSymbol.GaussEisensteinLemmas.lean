/-- The image of the map sending a nonzero natural number `x ≤ p / 2` to the absolute value
  of the integer in `(-p/2, p/2]` that is congruent to `a * x mod p` is the set
  of nonzero natural numbers `x` such that `x ≤ p / 2`. -/
theorem Ico_map_valMinAbs_natAbs_eq_Ico_map_id (p : ℕ) [hp : Fact p.Prime] (a : ZMod p)
    (hap : a ≠ 0) : ((Ico 1 (p / 2).succ).1.map fun (x : ℕ) => (a * x).valMinAbs.natAbs) =
    (Ico 1 (p / 2).succ).1.map fun a => a := by
  have he : ∀ {x}, x ∈ Ico 1 (p / 2).succ → x ≠ 0 ∧ x ≤ p / 2 := by
    simp +contextual [Nat.lt_succ_iff, Nat.succ_le_iff, pos_iff_ne_zero]
  have hep : ∀ {x}, x ∈ Ico 1 (p / 2).succ → x < p := fun hx =>
    lt_of_le_of_lt (he hx).2 (Nat.div_lt_self hp.1.pos (by decide))
  have hpe : ∀ {x}, x ∈ Ico 1 (p / 2).succ → ¬p ∣ x := fun hx hpx =>
    not_lt_of_ge (le_of_dvd (Nat.pos_of_ne_zero (he hx).1) hpx) (hep hx)
  have hmem : ∀ (x : ℕ) (_ : x ∈ Ico 1 (p / 2).succ),
      (a * x : ZMod p).valMinAbs.natAbs ∈ Ico 1 (p / 2).succ := by
    intro x hx
    simp [hap, CharP.cast_eq_zero_iff (ZMod p) p, hpe hx, Nat.lt_succ_iff, succ_le_iff,
      pos_iff_ne_zero, natAbs_valMinAbs_le _]
  have hsurj : ∀ (b : ℕ) (hb : b ∈ Ico 1 (p / 2).succ),
      ∃ x, ∃ _ : x ∈ Ico 1 (p / 2).succ, (a * x : ZMod p).valMinAbs.natAbs = b := by
    intro b hb
    refine ⟨(b / a : ZMod p).valMinAbs.natAbs, mem_Ico.mpr ⟨?_, ?_⟩, ?_⟩
    · apply Nat.pos_of_ne_zero
      simp only [div_eq_mul_inv, hap, CharP.cast_eq_zero_iff (ZMod p) p, hpe hb, not_false_iff,
        valMinAbs_eq_zero, inv_eq_zero, Int.natAbs_eq_zero, Ne, _root_.mul_eq_zero, or_self_iff]
    · apply lt_succ_of_le; apply natAbs_valMinAbs_le
    · rw [natCast_natAbs_valMinAbs]
      split_ifs
      · rw [mul_div_cancel₀ _ hap, valMinAbs_def_pos, val_cast_of_lt (hep hb),
          if_pos (le_of_lt_succ (mem_Ico.1 hb).2), Int.natAbs_ofNat]
      · rw [mul_neg, mul_div_cancel₀ _ hap, natAbs_valMinAbs_neg, valMinAbs_def_pos,
          val_cast_of_lt (hep hb), if_pos (le_of_lt_succ (mem_Ico.1 hb).2), Int.natAbs_ofNat]
  exact Multiset.map_eq_map_of_bij_of_nodup _ _ (Finset.nodup _) (Finset.nodup _)
    (fun x _ => (a * x : ZMod p).valMinAbs.natAbs) hmem
    (inj_on_of_surj_on_of_card_le _ hmem hsurj le_rfl) hsurj (fun _ _ => rfl)


private theorem gauss_lemma_aux₁ (p : ℕ) [Fact p.Prime] {a : ℤ} (hap : (a : ZMod p) ≠ 0) :
    (a ^ (p / 2) * (p / 2)! : ZMod p) =
     (-1 : ZMod p) ^ #{x ∈ Ico 1 (p / 2).succ | ¬ (a * x.cast : ZMod p).val ≤ p / 2} * (p / 2)! :=
  calc
    (a ^ (p / 2) * (p / 2)! : ZMod p) = ∏ x ∈ Ico 1 (p / 2).succ, a * x := by
      rw [prod_mul_distrib, ← prod_natCast, prod_Ico_id_eq_factorial, prod_const, card_Ico,
                              /-
                                p : Nat
                                inst✝ : Fact (Nat.Prime p)
                                a : Int
                                hap : Ne (↑a) 0
                                ⊢ Eq (HMul.hMul (HPow.hPow (↑a) (HDiv.hDiv p 2)) ↑(HDiv.hDiv p 2).factorial) ↑ …
                              -/
        Nat.add_one_sub_one]; simp
                              /-
                                🎉 no goals
                              -/
                                                                /-
                                                                  p : Nat
                                                                  inst✝ : Fact (Nat.Prime p)
                                                                  a : Int
                                                                  hap : Ne (↑a) 0
                                                                  ⊢ Eq (↑((Finset.Ico 1 (HDiv.hDiv p 2).succ).prod fun x => HMul.hMul a ↑x)) ((F …
                                                                -/
    _ = ∏ x ∈ Ico 1 (p / 2).succ, ↑((a * x : ZMod p).val) := by simp
                                                                /-
                                                                  🎉 no goals
                                                                -/
    _ = ∏ x ∈ Ico 1 (p / 2).succ, (if (a * x : ZMod p).val ≤ p / 2 then (1 : ZMod p) else -1) *
        (a * x : ZMod p).valMinAbs.natAbs :=
      (prod_congr rfl fun _ _ => by
        /-
          p : Nat
          inst✝ : Fact (Nat.Prime p)
          a : Int
          hap : Ne (↑a) 0
          x✝¹ : Nat
          x✝ : Membership.mem (Finset.Ico 1 (HDiv.hDiv p 2).succ) x✝¹
          ⊢ Eq (↑(HMul.hMul ↑a ↑x✝¹).val) (HMul.hMul (ite (LE.le (HMul.hMul ↑a ↑x✝¹).val …
        -/
        simp only [natCast_natAbs_valMinAbs]
        /-
          p : Nat
          inst✝ : Fact (Nat.Prime p)
          a : Int
          hap : Ne (↑a) 0
          x✝¹ : Nat
          x✝ : Membership.mem (Finset.Ico 1 (HDiv.hDiv p 2).succ) x✝¹
          ⊢ Eq (↑(HMul.hMul ↑a ↑x✝¹).val) (HMul.hMul (ite (LE.le (HMul.hMul ↑a ↑x✝¹).val …
        -/
                      /-
                        🎉 no goals
                      -/
        split_ifs <;> simp)
                      /-
                        🎉 no goals
                      -/
    _ = (-1 : ZMod p) ^ #{x ∈ Ico 1 (p / 2).succ | ¬(a * x.cast : ZMod p).val ≤ p / 2} *
          ∏ x ∈ Ico 1 (p / 2).succ, ↑((a * x : ZMod p).valMinAbs.natAbs) := by
      have :
          (∏ x ∈ Ico 1 (p / 2).succ, if (a * x : ZMod p).val ≤ p / 2 then (1 : ZMod p) else -1) =
          ∏ x ∈ Ico 1 (p / 2).succ with ¬(a * x.cast : ZMod p).val ≤ p / 2, -1 :=
        prod_bij_ne_one (fun x _ _ => x)
          (fun x => by split_ifs <;> (dsimp; simp_all))
          (fun _ _ _ _ _ _ => id) (fun b h _ => ⟨b, by simp_all [-not_le]⟩)
          (by intros; split_ifs at * <;> simp_all)
      /-
        p : Nat
        inst✝ : Fact (Nat.Prime p)
        a : Int
        hap : Ne (↑a) 0
        this : Eq ((Finset.Ico 1 (HDiv.hDiv p 2).succ).prod fun x => ite (LE.le (HMul. …
        ⊢ Eq ((Finset.Ico 1 (HDiv.hDiv p 2).succ).prod fun x => HMul.hMul (ite (LE.le  …
      -/
      rw [prod_mul_distrib, this, prod_const]
      /-
        🎉 no goals
      -/
    _ = (-1 : ZMod p) ^ #{x ∈ Ico 1 (p / 2).succ | ¬(a * x.cast : ZMod p).val ≤ p / 2} *
          (p / 2)! := by
      rw [← prod_natCast, Finset.prod_eq_multiset_prod,
        Ico_map_valMinAbs_natAbs_eq_Ico_map_id p a hap, ← Finset.prod_eq_multiset_prod,
        prod_Ico_id_eq_factorial]


theorem gauss_lemma_aux (p : ℕ) [hp : Fact p.Prime] {a : ℤ} (hap : (a : ZMod p) ≠ 0) :
    (a ^ (p / 2) : ZMod p) =
      ((-1) ^ #{x ∈ Ico 1 (p / 2).succ | p / 2 < (a * x.cast : ZMod p).val} :) :=
  (mul_left_inj' (show ((p / 2)! : ZMod p) ≠ 0 by
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      a : Int
      hap : Ne (↑a) 0
      ⊢ Ne (↑(HDiv.hDiv p 2).factorial) 0
    -/
    rw [Ne, CharP.cast_eq_zero_iff (ZMod p) p, hp.1.dvd_factorial, not_le]
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      a : Int
      hap : Ne (↑a) 0
      ⊢ LT.lt (HDiv.hDiv p 2) p
    -/
    exact Nat.div_lt_self hp.1.pos (by decide))).1 <| by
    /-
      🎉 no goals
    -/
      /-
        p : Nat
        hp : Fact (Nat.Prime p)
        a : Int
        hap : Ne (↑a) 0
        ⊢ Eq (HMul.hMul (HPow.hPow (↑a) (HDiv.hDiv p 2)) ↑(HDiv.hDiv p 2).factorial) ( …
      -/
      simpa using gauss_lemma_aux₁ p hap
      /-
        🎉 no goals
      -/


/-- **Gauss' lemma**. The Legendre symbol can be computed by considering the number of naturals less
  than `p/2` such that `(a * x) % p > p / 2`. -/
theorem gauss_lemma {p : ℕ} [h : Fact p.Prime] {a : ℤ} (hp : p ≠ 2) (ha0 : (a : ZMod p) ≠ 0) :
    legendreSym p a = (-1) ^ #{x ∈ Ico 1 (p / 2).succ | p / 2 < (a * x.cast : ZMod p).val} := by
  /-
    p : Nat
    h : Fact (Nat.Prime p)
    a : Int
    hp : Ne p 2
    ha0 : Ne (↑a) 0
    ⊢ Eq (legendreSym p a) (HPow.hPow (-1) (Finset.filter (fun x => LT.lt (HDiv.hD …
  -/
  replace hp : Odd p := h.out.odd_of_ne_two hp
  have : (legendreSym p a : ZMod p) =
      (((-1) ^ #{x ∈ Ico 1 (p / 2).succ | p / 2 < (a * x.cast : ZMod p).val} : ℤ) : ZMod p) := by
    rw [legendreSym.eq_pow, gauss_lemma_aux p ha0]
  /-
    p : Nat
    h : Fact (Nat.Prime p)
    a : Int
    ha0 : Ne (↑a) 0
    hp : Odd p
    this : Eq ↑(legendreSym p a) ↑(HPow.hPow (-1) (Finset.filter (fun x => LT.lt ( …
    ⊢ Eq (legendreSym p a) (HPow.hPow (-1) (Finset.filter (fun x => LT.lt (HDiv.hD …
  -/
  cases legendreSym.eq_one_or_neg_one p ha0 <;>
  /-
    case inl
    p : Nat
    h : Fact (Nat.Prime p)
    a : Int
    ha0 : Ne (↑a) 0
    hp : Odd p
    this : Eq ↑(legendreSym p a) ↑(HPow.hPow (-1) (Finset.filter (fun x => LT.lt ( …
    h✝ : Eq (legendreSym p a) 1
    ⊢ Eq (legendreSym p a) (HPow.hPow (-1) (Finset.filter (fun x => LT.lt (HDiv.hD …
  -/
  cases neg_one_pow_eq_or ℤ #{x ∈ Ico 1 (p / 2).succ | p / 2 < (a * x.cast : ZMod p).val} <;>
  /-
    case inl.inl
    p : Nat
    h : Fact (Nat.Prime p)
    a : Int
    ha0 : Ne (↑a) 0
    hp : Odd p
    this : Eq ↑(legendreSym p a) ↑(HPow.hPow (-1) (Finset.filter (fun x => LT.lt ( …
    h✝¹ : Eq (legendreSym p a) 1
    h✝ : Eq (HPow.hPow (-1) (Finset.filter (fun x => LT.lt (HDiv.hDiv p 2) (HMul.h …
    ⊢ Eq (legendreSym p a) (HPow.hPow (-1) (Finset.filter (fun x => LT.lt (HDiv.hD …
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  simp_all [ne_neg_self hp one_ne_zero, (ne_neg_self hp one_ne_zero).symm]
  /-
    🎉 no goals
  -/


private theorem eisenstein_lemma_aux₁ (p : ℕ) [Fact p.Prime] [hp2 : Fact (p % 2 = 1)] {a : ℕ}
    (hap : (a : ZMod p) ≠ 0) :
    ((∑ x ∈ Ico 1 (p / 2).succ, a * x : ℕ) : ZMod 2) =
      #{x ∈ Ico 1 (p / 2).succ | p / 2 < (a * x.cast : ZMod p).val} +
        ∑ x ∈ Ico 1 (p / 2).succ, x + (∑ x ∈ Ico 1 (p / 2).succ, a * x / p : ℕ) :=
  have hp2 : (p : ZMod 2) = (1 : ℕ) := (eq_iff_modEq_nat _).2 hp2.1
  calc
    ((∑ x ∈ Ico 1 (p / 2).succ, a * x : ℕ) : ZMod 2) =
        ((∑ x ∈ Ico 1 (p / 2).succ, (a * x % p + p * (a * x / p)) : ℕ) : ZMod 2) := by
      /-
        p : Nat
        inst✝ : Fact (Nat.Prime p)
        hp2✝ : Fact (Eq (HMod.hMod p 2) 1)
        a : Nat
        hap : Ne (↑a) 0
        hp2 : Eq ↑p ↑1
        ⊢ Eq ↑((Finset.Ico 1 (HDiv.hDiv p 2).succ).sum fun x => HMul.hMul a x) ↑((Fins …
      -/
      simp only [mod_add_div]
      /-
        🎉 no goals
      -/
    _ = (∑ x ∈ Ico 1 (p / 2).succ, ((a * x : ℕ) : ZMod p).val : ℕ) +
        (∑ x ∈ Ico 1 (p / 2).succ, a * x / p : ℕ) := by
      /-
        p : Nat
        inst✝ : Fact (Nat.Prime p)
        hp2✝ : Fact (Eq (HMod.hMod p 2) 1)
        a : Nat
        hap : Ne (↑a) 0
        hp2 : Eq ↑p ↑1
        ⊢ Eq (↑((Finset.Ico 1 (HDiv.hDiv p 2).succ).sum fun x => HAdd.hAdd (HMod.hMod  …
      -/
      simp only [val_natCast]
      /-
        p : Nat
        inst✝ : Fact (Nat.Prime p)
        hp2✝ : Fact (Eq (HMod.hMod p 2) 1)
        a : Nat
        hap : Ne (↑a) 0
        hp2 : Eq ↑p ↑1
        ⊢ Eq (↑((Finset.Ico 1 (HDiv.hDiv p 2).succ).sum fun x => HAdd.hAdd (HMod.hMod  …
      -/
      simp [sum_add_distrib, ← mul_sum, Nat.cast_add, Nat.cast_mul, Nat.cast_sum, hp2]
      /-
        🎉 no goals
      -/
    _ = _ :=
      congr_arg₂ (· + ·)
        (calc
          ((∑ x ∈ Ico 1 (p / 2).succ, ((a * x : ℕ) : ZMod p).val : ℕ) : ZMod 2) =
              ∑ x ∈ Ico 1 (p / 2).succ, (((a * x : ZMod p).valMinAbs +
                if (a * x : ZMod p).val ≤ p / 2 then 0 else p : ℤ) : ZMod 2) := by
            /-
              p : Nat
              inst✝ : Fact (Nat.Prime p)
              hp2✝ : Fact (Eq (HMod.hMod p 2) 1)
              a : Nat
              hap : Ne (↑a) 0
              hp2 : Eq ↑p ↑1
              ⊢ Eq (↑((Finset.Ico 1 (HDiv.hDiv p 2).succ).sum fun x => (↑(HMul.hMul a x)).va …
            -/
            simp only [(val_eq_ite_valMinAbs _).symm]; simp [Nat.cast_sum]
                                                       /-
                                                         🎉 no goals
                                                       -/
          _ = #{x ∈ Ico 1 (p / 2).succ | p / 2 < (a * x.cast : ZMod p).val} +
              (∑ x ∈ Ico 1 (p / 2).succ, (a * x.cast : ZMod p).valMinAbs.natAbs : ℕ) := by
            /-
              p : Nat
              inst✝ : Fact (Nat.Prime p)
              hp2✝ : Fact (Eq (HMod.hMod p 2) 1)
              a : Nat
              hap : Ne (↑a) 0
              hp2 : Eq ↑p ↑1
              ⊢ Eq ((Finset.Ico 1 (HDiv.hDiv p 2).succ).sum fun x => ↑(HAdd.hAdd (HMul.hMul  …
            -/
            simp [add_comm, sum_add_distrib, Finset.sum_ite, hp2, Nat.cast_sum]
            /-
              🎉 no goals
            -/
          _ = _ := by
            rw [Finset.sum_eq_multiset_sum, Ico_map_valMinAbs_natAbs_eq_Ico_map_id p a hap, ←
              Finset.sum_eq_multiset_sum])
        rfl


theorem eisenstein_lemma_aux (p : ℕ) [Fact p.Prime] [Fact (p % 2 = 1)] {a : ℕ} (ha2 : a % 2 = 1)
    (hap : (a : ZMod p) ≠ 0) :
    #{x ∈ Ico 1 (p / 2).succ | p / 2 < (a * x.cast : ZMod p).val} ≡
      ∑ x ∈ Ico 1 (p / 2).succ, x * a / p [MOD 2] :=
  have ha2 : (a : ZMod 2) = (1 : ℕ) := (eq_iff_modEq_nat _).2 ha2
  (eq_iff_modEq_nat 2).1 <| sub_eq_zero.1 <| by
    simpa [add_left_comm, sub_eq_add_neg, ← mul_sum, mul_comm, ha2, Nat.cast_sum,
      add_neg_eq_iff_eq_add.symm, neg_eq_self_mod_two, add_assoc] using
      Eq.symm (eisenstein_lemma_aux₁ p hap)


theorem div_eq_filter_card {a b c : ℕ} (hb0 : 0 < b) (hc : a / b ≤ c) :
    a / b = #{x ∈ Ico 1 c.succ | x * b ≤ a} :=
  calc
                                        /-
                                          a b c : Nat
                                          hb0 : LT.lt 0 b
                                          hc : LE.le (HDiv.hDiv a b) c
                                          ⊢ Eq (HDiv.hDiv a b) (Finset.Ico 1 (HDiv.hDiv a b).succ).card
                                        -/
    a / b = #(Ico 1 (a / b).succ) := by simp
                                        /-
                                          🎉 no goals
                                        -/
    _ = #{x ∈ Ico 1 c.succ | x * b ≤ a} :=
      congr_arg _ <| Finset.ext fun x => by
        /-
          a b c : Nat
          hb0 : LT.lt 0 b
          hc : LE.le (HDiv.hDiv a b) c
          x : Nat
          ⊢ Iff (Membership.mem (Finset.Ico 1 (HDiv.hDiv a b).succ) x) (Membership.mem ( …
        -/
        have : x * b ≤ a → x ≤ c := fun h => le_trans (by rwa [le_div_iff_mul_le hb0]) hc
        /-
          a b c : Nat
          hb0 : LT.lt 0 b
          hc : LE.le (HDiv.hDiv a b) c
          x : Nat
          this : LE.le (HMul.hMul x b) a → LE.le x c
          ⊢ Iff (Membership.mem (Finset.Ico 1 (HDiv.hDiv a b).succ) x) (Membership.mem ( …
        -/
        simp [Nat.lt_succ_iff, le_div_iff_mul_le hb0]; tauto
                                                       /-
                                                         🎉 no goals
                                                       -/


/-- The given sum is the number of integer points in the triangle formed by the diagonal of the
  rectangle `(0, p/2) × (0, q/2)`. -/
private theorem sum_Ico_eq_card_lt {p q : ℕ} :
    ∑ a ∈ Ico 1 (p / 2).succ, a * q / p =
      #{x ∈ Ico 1 (p / 2).succ ×ˢ Ico 1 (q / 2).succ | x.2 * p ≤ x.1 * q} :=
                         /-
                           p q : Nat
                           hp0 : Eq p 0
                           ⊢ Eq ((Finset.Ico 1 (HDiv.hDiv p 2).succ).sum fun a => HDiv.hDiv (HMul.hMul a  …
                         -/
  if hp0 : p = 0 then by simp [hp0, Finset.ext_iff]
                         /-
                           🎉 no goals
                         -/
  else
    calc
      ∑ a ∈ Ico 1 (p / 2).succ, a * q / p =
          ∑ a ∈ Ico 1 (p / 2).succ, #{x ∈ Ico 1 (q / 2).succ | x * p ≤ a * q} :=
        Finset.sum_congr rfl fun x hx => div_eq_filter_card (Nat.pos_of_ne_zero hp0) <|
          calc
                                            /-
                                              p q : Nat
                                              hp0 : Not (Eq p 0)
                                              x : Nat
                                              hx : Membership.mem (Finset.Ico 1 (HDiv.hDiv p 2).succ) x
                                              ⊢ LE.le (HDiv.hDiv (HMul.hMul x q) p) (HDiv.hDiv (HMul.hMul (HDiv.hDiv p 2) q) …
                                            -/
            x * q / p ≤ p / 2 * q / p := by have := le_of_lt_succ (mem_Ico.mp hx).2; gcongr
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/
            _ ≤ _ := Nat.div_mul_div_le_div _ _ _
      _ = _ := by
        /-
          p q : Nat
          hp0 : Not (Eq p 0)
          ⊢ Eq ((Finset.Ico 1 (HDiv.hDiv p 2).succ).sum fun a => (Finset.filter (fun x = …
        -/
        rw [← card_sigma]
        exact card_nbij' (fun a ↦ ⟨a.1, a.2⟩) (fun a ↦ ⟨a.1, a.2⟩)
          (by simp +contextual only [mem_filter, mem_sigma, and_self_iff,
            forall_true_iff, mem_product])
          (by simp +contextual only [mem_filter, mem_sigma, and_self_iff,
            forall_true_iff, mem_product]) (fun _ _ ↦ rfl) (fun _ _ ↦ rfl)


/-- Each of the sums in this lemma is the cardinality of the set of integer points in each of the
  two triangles formed by the diagonal of the rectangle `(0, p/2) × (0, q/2)`. Adding them
  gives the number of points in the rectangle. -/
theorem sum_mul_div_add_sum_mul_div_eq_mul (p q : ℕ) [hp : Fact p.Prime] (hq0 : (q : ZMod p) ≠ 0) :
    ∑ a ∈ Ico 1 (p / 2).succ, a * q / p + ∑ a ∈ Ico 1 (q / 2).succ, a * p / q =
    p / 2 * (q / 2) := by
  have hswap :
    #{x ∈ Ico 1 (q / 2).succ ×ˢ Ico 1 (p / 2).succ | x.2 * q ≤ x.1 * p} =
      #{x ∈ Ico 1 (p / 2).succ ×ˢ Ico 1 (q / 2).succ | x.1 * q ≤ x.2 * p} :=
    card_equiv (Equiv.prodComm _ _)
      (fun ⟨_, _⟩ => by
        simp +contextual only [mem_filter, and_self_iff, Prod.swap_prod_mk,
          forall_true_iff, mem_product, Equiv.prodComm_apply, and_assoc, and_left_comm])
  have hdisj :
    Disjoint {x ∈ Ico 1 (p / 2).succ ×ˢ Ico 1 (q / 2).succ | x.2 * p ≤ x.1 * q}
      {x ∈ Ico 1 (p / 2).succ ×ˢ Ico 1 (q / 2).succ | x.1 * q ≤ x.2 * p} := by
    apply disjoint_filter.2 fun x hx hpq hqp => ?_
    have hxp : x.1 < p := lt_of_le_of_lt
      (show x.1 ≤ p / 2 by simp_all only [Nat.lt_succ_iff, mem_Ico, mem_product])
      (Nat.div_lt_self hp.1.pos (by decide))
    have : (x.1 : ZMod p) = 0 := by
      simpa [hq0] using congr_arg ((↑) : ℕ → ZMod p) (le_antisymm hpq hqp)
    apply_fun ZMod.val at this
    rw [val_cast_of_lt hxp, val_zero] at this
    simp only [this, nonpos_iff_eq_zero, mem_Ico, one_ne_zero, false_and, mem_product] at hx
  have hunion :
      {x ∈ Ico 1 (p / 2).succ ×ˢ Ico 1 (q / 2).succ | x.2 * p ≤ x.1 * q} ∪
        {x ∈ Ico 1 (p / 2).succ ×ˢ Ico 1 (q / 2).succ | x.1 * q ≤ x.2 * p} =
      Ico 1 (p / 2).succ ×ˢ Ico 1 (q / 2).succ :=
    Finset.ext fun x => by
      have := le_total (x.2 * p) (x.1 * q)
      simp only [mem_union, mem_filter, mem_Ico, mem_product]
      tauto
  rw [sum_Ico_eq_card_lt, sum_Ico_eq_card_lt, hswap, ← card_union_of_disjoint hdisj, hunion,
    card_product]
  /-
    p q : Nat
    hp : Fact (Nat.Prime p)
    hq0 : Ne (↑q) 0
    hswap : Eq (Finset.filter (fun x => LE.le (HMul.hMul x.2 q) (HMul.hMul x.1 p)) …
    hdisj : Disjoint (Finset.filter (fun x => LE.le (HMul.hMul x.2 p) (HMul.hMul x …
    hunion : Eq (Union.union (Finset.filter (fun x => LE.le (HMul.hMul x.2 p) (HMu …
    ⊢ Eq (HMul.hMul (Finset.Ico 1 (HDiv.hDiv p 2).succ).card (Finset.Ico 1 (HDiv.h …
  -/
  simp only [card_Ico, tsub_zero, succ_sub_succ_eq_sub]
  /-
    🎉 no goals
  -/


/-- **Eisenstein's lemma** -/
theorem eisenstein_lemma {p : ℕ} [Fact p.Prime] (hp : p ≠ 2) {a : ℕ} (ha1 : a % 2 = 1)
    (ha0 : (a : ZMod p) ≠ 0) : legendreSym p a = (-1) ^ ∑ x ∈ Ico 1 (p / 2).succ, x * a / p := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    hp : Ne p 2
    a : Nat
    ha1 : Eq (HMod.hMod a 2) 1
    ha0 : Ne (↑a) 0
    ⊢ Eq (legendreSym p ↑a) (HPow.hPow (-1) ((Finset.Ico 1 (HDiv.hDiv p 2).succ).s …
  -/
  haveI hp' : Fact (p % 2 = 1) := ⟨Nat.Prime.mod_two_eq_one_iff_ne_two.mpr hp⟩
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    hp : Ne p 2
    a : Nat
    ha1 : Eq (HMod.hMod a 2) 1
    ha0 : Ne (↑a) 0
    hp' : Fact (Eq (HMod.hMod p 2) 1)
    ⊢ Eq (legendreSym p ↑a) (HPow.hPow (-1) ((Finset.Ico 1 (HDiv.hDiv p 2).succ).s …
  -/
  have ha0' : ((a : ℤ) : ZMod p) ≠ 0 := by norm_cast
  rw [neg_one_pow_eq_pow_mod_two, gauss_lemma hp ha0', neg_one_pow_eq_pow_mod_two,
    (by norm_cast : ((a : ℤ) : ZMod p) = (a : ZMod p)),
    show _ = _ from eisenstein_lemma_aux p ha1 ha0]


