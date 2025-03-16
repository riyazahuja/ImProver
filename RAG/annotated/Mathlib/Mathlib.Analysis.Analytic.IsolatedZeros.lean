theorem hasSum_at_zero (a : ℕ → E) : HasSum (fun n => (0 : 𝕜) ^ n • a n) (a 0) := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    a : Nat → E
    ⊢ HasSum (fun n => HSMul.hSMul (HPow.hPow 0 n) (a n)) (a 0)
  -/
                                                   /-
                                                     🎉 no goals
                                                   -/
  convert hasSum_single (α := E) 0 fun b h ↦ _ <;> simp [*]
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem exists_hasSum_smul_of_apply_eq_zero (hs : HasSum (fun m => z ^ m • a m) s)
    (ha : ∀ k < n, a k = 0) : ∃ t : E, z ^ n • t = s ∧ HasSum (fun m => z ^ m • a (m + n)) t := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    s : E
    n : Nat
    z : 𝕜
    a : Nat → E
    hs : HasSum (fun m => HSMul.hSMul (HPow.hPow z m) (a m)) s
    ha : ∀ (k : Nat), LT.lt k n → Eq (a k) 0
    ⊢ Exists fun t => And (Eq (HSMul.hSMul (HPow.hPow z n) t) s) (HasSum (fun m => …
  -/
  obtain rfl | hn := n.eq_zero_or_pos
    /-
      case inl
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      s : E
      z : 𝕜
      a : Nat → E
      hs : HasSum (fun m => HSMul.hSMul (HPow.hPow z m) (a m)) s
      ha : ∀ (k : Nat), LT.lt k 0 → Eq (a k) 0
      ⊢ Exists fun t => And (Eq (HSMul.hSMul (HPow.hPow z 0) t) s) (HasSum (fun m => …
    -/
  · simpa
    /-
      🎉 no goals
    -/
  /-
    case inr
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    s : E
    n : Nat
    z : 𝕜
    a : Nat → E
    hs : HasSum (fun m => HSMul.hSMul (HPow.hPow z m) (a m)) s
    ha : ∀ (k : Nat), LT.lt k n → Eq (a k) 0
    hn : GT.gt n 0
    ⊢ Exists fun t => And (Eq (HSMul.hSMul (HPow.hPow z n) t) s) (HasSum (fun m => …
  -/
  by_cases h : z = 0
    /-
      case pos
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      s : E
      n : Nat
      z : 𝕜
      a : Nat → E
      hs : HasSum (fun m => HSMul.hSMul (HPow.hPow z m) (a m)) s
      ha : ∀ (k : Nat), LT.lt k n → Eq (a k) 0
      hn : GT.gt n 0
      h : Eq z 0
      ⊢ Exists fun t => And (Eq (HSMul.hSMul (HPow.hPow z n) t) s) (HasSum (fun m => …
    -/
  · have : s = 0 := hs.unique (by simpa [ha 0 hn, h] using hasSum_at_zero a)
    /-
      case pos
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      s : E
      n : Nat
      z : 𝕜
      a : Nat → E
      hs : HasSum (fun m => HSMul.hSMul (HPow.hPow z m) (a m)) s
      ha : ∀ (k : Nat), LT.lt k n → Eq (a k) 0
      hn : GT.gt n 0
      h : Eq z 0
      this : Eq s 0
      ⊢ Exists fun t => And (Eq (HSMul.hSMul (HPow.hPow z n) t) s) (HasSum (fun m => …
    -/
    exact ⟨a n, by simp [h, hn.ne', this], by simpa [h] using hasSum_at_zero fun m => a (m + n)⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      s : E
      n : Nat
      z : 𝕜
      a : Nat → E
      hs : HasSum (fun m => HSMul.hSMul (HPow.hPow z m) (a m)) s
      ha : ∀ (k : Nat), LT.lt k n → Eq (a k) 0
      hn : GT.gt n 0
      h : Not (Eq z 0)
      ⊢ Exists fun t => And (Eq (HSMul.hSMul (HPow.hPow z n) t) s) (HasSum (fun m => …
    -/
  · refine ⟨(z ^ n)⁻¹ • s, by field_simp [smul_smul], ?_⟩
    have h1 : ∑ i ∈ Finset.range n, z ^ i • a i = 0 :=
      Finset.sum_eq_zero fun k hk => by simp [ha k (Finset.mem_range.mp hk)]
    have h2 : HasSum (fun m => z ^ (m + n) • a (m + n)) s := by
      simpa [h1] using (hasSum_nat_add_iff' n).mpr hs
    /-
      case neg
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      s : E
      n : Nat
      z : 𝕜
      a : Nat → E
      hs : HasSum (fun m => HSMul.hSMul (HPow.hPow z m) (a m)) s
      ha : ∀ (k : Nat), LT.lt k n → Eq (a k) 0
      hn : GT.gt n 0
      h : Not (Eq z 0)
      h1 : Eq ((Finset.range n).sum fun i => HSMul.hSMul (HPow.hPow z i) (a i)) 0
      h2 : HasSum (fun m => HSMul.hSMul (HPow.hPow z (HAdd.hAdd m n)) (a (HAdd.hAdd  …
      ⊢ HasSum (fun m => HSMul.hSMul (HPow.hPow z m) (a (HAdd.hAdd m n))) (HSMul.hSM …
    -/
    convert h2.const_smul (z⁻¹ ^ n) using 1
      /-
        case h.e'_5
        𝕜 : Type u_1
        inst✝² : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace 𝕜 E
        s : E
        n : Nat
        z : 𝕜
        a : Nat → E
        hs : HasSum (fun m => HSMul.hSMul (HPow.hPow z m) (a m)) s
        ha : ∀ (k : Nat), LT.lt k n → Eq (a k) 0
        hn : GT.gt n 0
        h : Not (Eq z 0)
        h1 : Eq ((Finset.range n).sum fun i => HSMul.hSMul (HPow.hPow z i) (a i)) 0
        h2 : HasSum (fun m => HSMul.hSMul (HPow.hPow z (HAdd.hAdd m n)) (a (HAdd.hAdd  …
        ⊢ Eq (fun m => HSMul.hSMul (HPow.hPow z m) (a (HAdd.hAdd m n))) fun i => HSMul …
      -/
    · field_simp [pow_add, smul_smul]
      /-
        🎉 no goals
      -/
      /-
        case h.e'_6
        𝕜 : Type u_1
        inst✝² : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace 𝕜 E
        s : E
        n : Nat
        z : 𝕜
        a : Nat → E
        hs : HasSum (fun m => HSMul.hSMul (HPow.hPow z m) (a m)) s
        ha : ∀ (k : Nat), LT.lt k n → Eq (a k) 0
        hn : GT.gt n 0
        h : Not (Eq z 0)
        h1 : Eq ((Finset.range n).sum fun i => HSMul.hSMul (HPow.hPow z i) (a i)) 0
        h2 : HasSum (fun m => HSMul.hSMul (HPow.hPow z (HAdd.hAdd m n)) (a (HAdd.hAdd  …
        ⊢ Eq (HSMul.hSMul (Inv.inv (HPow.hPow z n)) s) (HSMul.hSMul (HPow.hPow (Inv.in …
      -/
    · simp only [inv_pow]
      /-
        🎉 no goals
      -/


theorem has_fpower_series_dslope_fslope (hp : HasFPowerSeriesAt f p z₀) :
    HasFPowerSeriesAt (dslope f z₀) p.fslope z₀ := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    p : FormalMultilinearSeries 𝕜 𝕜 E
    f : 𝕜 → E
    z₀ : 𝕜
    hp : HasFPowerSeriesAt f p z₀
    ⊢ HasFPowerSeriesAt (dslope f z₀) p.fslope z₀
  -/
  have hpd : deriv f z₀ = p.coeff 1 := hp.deriv
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    p : FormalMultilinearSeries 𝕜 𝕜 E
    f : 𝕜 → E
    z₀ : 𝕜
    hp : HasFPowerSeriesAt f p z₀
    hpd : Eq (deriv f z₀) (p.coeff 1)
    ⊢ HasFPowerSeriesAt (dslope f z₀) p.fslope z₀
  -/
  have hp0 : p.coeff 0 = f z₀ := hp.coeff_zero 1
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    p : FormalMultilinearSeries 𝕜 𝕜 E
    f : 𝕜 → E
    z₀ : 𝕜
    hp : HasFPowerSeriesAt f p z₀
    hpd : Eq (deriv f z₀) (p.coeff 1)
    hp0 : Eq (p.coeff 0) (f z₀)
    ⊢ HasFPowerSeriesAt (dslope f z₀) p.fslope z₀
  -/
  simp only [hasFPowerSeriesAt_iff, apply_eq_pow_smul_coeff, coeff_fslope] at hp ⊢
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    p : FormalMultilinearSeries 𝕜 𝕜 E
    f : 𝕜 → E
    z₀ : 𝕜
    hpd : Eq (deriv f z₀) (p.coeff 1)
    hp0 : Eq (p.coeff 0) (f z₀)
    hp : Filter.Eventually (fun z => HasSum (fun n => HSMul.hSMul (HPow.hPow z n)  …
    ⊢ Filter.Eventually (fun z => HasSum (fun n => HSMul.hSMul (HPow.hPow z n) (p. …
  -/
  refine hp.mono fun x hx => ?_
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    p : FormalMultilinearSeries 𝕜 𝕜 E
    f : 𝕜 → E
    z₀ : 𝕜
    hpd : Eq (deriv f z₀) (p.coeff 1)
    hp0 : Eq (p.coeff 0) (f z₀)
    hp : Filter.Eventually (fun z => HasSum (fun n => HSMul.hSMul (HPow.hPow z n)  …
    x : 𝕜
    hx : HasSum (fun n => HSMul.hSMul (HPow.hPow x n) (p.coeff n)) (f (HAdd.hAdd z …
    ⊢ HasSum (fun n => HSMul.hSMul (HPow.hPow x n) (p.coeff (HAdd.hAdd n 1))) (dsl …
  -/
  by_cases h : x = 0
    /-
      case pos
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      p : FormalMultilinearSeries 𝕜 𝕜 E
      f : 𝕜 → E
      z₀ : 𝕜
      hpd : Eq (deriv f z₀) (p.coeff 1)
      hp0 : Eq (p.coeff 0) (f z₀)
      hp : Filter.Eventually (fun z => HasSum (fun n => HSMul.hSMul (HPow.hPow z n)  …
      x : 𝕜
      hx : HasSum (fun n => HSMul.hSMul (HPow.hPow x n) (p.coeff n)) (f (HAdd.hAdd z …
      h : Eq x 0
      ⊢ HasSum (fun n => HSMul.hSMul (HPow.hPow x n) (p.coeff (HAdd.hAdd n 1))) (dsl …
    -/
                                                      /-
                                                        🎉 no goals
                                                      -/
  · convert hasSum_single (α := E) 0 _ <;> intros <;> simp [*]
                                                      /-
                                                        🎉 no goals
                                                      -/
    /-
      case neg
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      p : FormalMultilinearSeries 𝕜 𝕜 E
      f : 𝕜 → E
      z₀ : 𝕜
      hpd : Eq (deriv f z₀) (p.coeff 1)
      hp0 : Eq (p.coeff 0) (f z₀)
      hp : Filter.Eventually (fun z => HasSum (fun n => HSMul.hSMul (HPow.hPow z n)  …
      x : 𝕜
      hx : HasSum (fun n => HSMul.hSMul (HPow.hPow x n) (p.coeff n)) (f (HAdd.hAdd z …
      h : Not (Eq x 0)
      ⊢ HasSum (fun n => HSMul.hSMul (HPow.hPow x n) (p.coeff (HAdd.hAdd n 1))) (dsl …
    -/
  · have hxx : ∀ n : ℕ, x⁻¹ * x ^ (n + 1) = x ^ n := fun n => by field_simp [h, _root_.pow_succ]
    suffices HasSum (fun n => x⁻¹ • x ^ (n + 1) • p.coeff (n + 1)) (x⁻¹ • (f (z₀ + x) - f z₀)) by
      simpa [dslope, slope, h, smul_smul, hxx] using this
    /-
      case neg
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      p : FormalMultilinearSeries 𝕜 𝕜 E
      f : 𝕜 → E
      z₀ : 𝕜
      hpd : Eq (deriv f z₀) (p.coeff 1)
      hp0 : Eq (p.coeff 0) (f z₀)
      hp : Filter.Eventually (fun z => HasSum (fun n => HSMul.hSMul (HPow.hPow z n)  …
      x : 𝕜
      hx : HasSum (fun n => HSMul.hSMul (HPow.hPow x n) (p.coeff n)) (f (HAdd.hAdd z …
      h : Not (Eq x 0)
      hxx : ∀ (n : Nat), Eq (HMul.hMul (Inv.inv x) (HPow.hPow x (HAdd.hAdd n 1))) (H …
      ⊢ HasSum (fun n => HSMul.hSMul (Inv.inv x) (HSMul.hSMul (HPow.hPow x (HAdd.hAd …
    -/
    simpa [hp0] using ((hasSum_nat_add_iff' 1).mpr hx).const_smul x⁻¹
    /-
      🎉 no goals
    -/


theorem has_fpower_series_iterate_dslope_fslope (n : ℕ) (hp : HasFPowerSeriesAt f p z₀) :
    HasFPowerSeriesAt ((swap dslope z₀)^[n] f) (fslope^[n] p) z₀ := by
  induction n generalizing f p with
  | zero => exact hp
  | succ n ih => simpa using ih (has_fpower_series_dslope_fslope hp)


theorem iterate_dslope_fslope_ne_zero (hp : HasFPowerSeriesAt f p z₀) (h : p ≠ 0) :
    (swap dslope z₀)^[p.order] f z₀ ≠ 0 := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    p : FormalMultilinearSeries 𝕜 𝕜 E
    f : 𝕜 → E
    z₀ : 𝕜
    hp : HasFPowerSeriesAt f p z₀
    h : Ne p 0
    ⊢ Ne (Nat.iterate (Function.swap dslope z₀) p.order f z₀) 0
  -/
  rw [← coeff_zero (has_fpower_series_iterate_dslope_fslope p.order hp) 1]
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    p : FormalMultilinearSeries 𝕜 𝕜 E
    f : 𝕜 → E
    z₀ : 𝕜
    hp : HasFPowerSeriesAt f p z₀
    h : Ne p 0
    ⊢ Ne ((Nat.iterate FormalMultilinearSeries.fslope p.order p 0) 1) 0
  -/
  simpa [coeff_eq_zero] using apply_order_ne_zero h
  /-
    🎉 no goals
  -/


theorem eq_pow_order_mul_iterate_dslope (hp : HasFPowerSeriesAt f p z₀) :
    ∀ᶠ z in 𝓝 z₀, f z = (z - z₀) ^ p.order • (swap dslope z₀)^[p.order] f z := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    p : FormalMultilinearSeries 𝕜 𝕜 E
    f : 𝕜 → E
    z₀ : 𝕜
    hp : HasFPowerSeriesAt f p z₀
    ⊢ Filter.Eventually (fun z => Eq (f z) (HSMul.hSMul (HPow.hPow (HSub.hSub z z₀ …
  -/
  have hq := hasFPowerSeriesAt_iff'.mp (has_fpower_series_iterate_dslope_fslope p.order hp)
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    p : FormalMultilinearSeries 𝕜 𝕜 E
    f : 𝕜 → E
    z₀ : 𝕜
    hp : HasFPowerSeriesAt f p z₀
    hq : Filter.Eventually (fun z => HasSum (fun n => HSMul.hSMul (HPow.hPow (HSub …
    ⊢ Filter.Eventually (fun z => Eq (f z) (HSMul.hSMul (HPow.hPow (HSub.hSub z z₀ …
  -/
  filter_upwards [hq, hasFPowerSeriesAt_iff'.mp hp] with x hx1 hx2
  have : ∀ k < p.order, p.coeff k = 0 := fun k hk => by
    simpa [coeff_eq_zero] using apply_eq_zero_of_lt_order hk
  /-
    case h
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    p : FormalMultilinearSeries 𝕜 𝕜 E
    f : 𝕜 → E
    z₀ : 𝕜
    hp : HasFPowerSeriesAt f p z₀
    hq : Filter.Eventually (fun z => HasSum (fun n => HSMul.hSMul (HPow.hPow (HSub …
    x : 𝕜
    hx1 : HasSum (fun n => HSMul.hSMul (HPow.hPow (HSub.hSub x z₀) n) ((Nat.iterat …
    hx2 : HasSum (fun n => HSMul.hSMul (HPow.hPow (HSub.hSub x z₀) n) (p.coeff n)) …
    this : ∀ (k : Nat), LT.lt k p.order → Eq (p.coeff k) 0
    ⊢ Eq (f x) (HSMul.hSMul (HPow.hPow (HSub.hSub x z₀) p.order) (Nat.iterate (Fun …
  -/
  obtain ⟨s, hs1, hs2⟩ := HasSum.exists_hasSum_smul_of_apply_eq_zero hx2 this
  /-
    case h.intro.intro
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    p : FormalMultilinearSeries 𝕜 𝕜 E
    f : 𝕜 → E
    z₀ : 𝕜
    hp : HasFPowerSeriesAt f p z₀
    hq : Filter.Eventually (fun z => HasSum (fun n => HSMul.hSMul (HPow.hPow (HSub …
    x : 𝕜
    hx1 : HasSum (fun n => HSMul.hSMul (HPow.hPow (HSub.hSub x z₀) n) ((Nat.iterat …
    hx2 : HasSum (fun n => HSMul.hSMul (HPow.hPow (HSub.hSub x z₀) n) (p.coeff n)) …
    this : ∀ (k : Nat), LT.lt k p.order → Eq (p.coeff k) 0
    s : E
    hs1 : Eq (HSMul.hSMul (HPow.hPow (HSub.hSub x z₀) p.order) s) (f x)
    hs2 : HasSum (fun m => HSMul.hSMul (HPow.hPow (HSub.hSub x z₀) m) (p.coeff (HA …
    ⊢ Eq (f x) (HSMul.hSMul (HPow.hPow (HSub.hSub x z₀) p.order) (Nat.iterate (Fun …
  -/
  convert hs1.symm
  /-
    case h.e'_3.h.e'_6
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    p : FormalMultilinearSeries 𝕜 𝕜 E
    f : 𝕜 → E
    z₀ : 𝕜
    hp : HasFPowerSeriesAt f p z₀
    hq : Filter.Eventually (fun z => HasSum (fun n => HSMul.hSMul (HPow.hPow (HSub …
    x : 𝕜
    hx1 : HasSum (fun n => HSMul.hSMul (HPow.hPow (HSub.hSub x z₀) n) ((Nat.iterat …
    hx2 : HasSum (fun n => HSMul.hSMul (HPow.hPow (HSub.hSub x z₀) n) (p.coeff n)) …
    this : ∀ (k : Nat), LT.lt k p.order → Eq (p.coeff k) 0
    s : E
    hs1 : Eq (HSMul.hSMul (HPow.hPow (HSub.hSub x z₀) p.order) s) (f x)
    hs2 : HasSum (fun m => HSMul.hSMul (HPow.hPow (HSub.hSub x z₀) m) (p.coeff (HA …
    ⊢ Eq (Nat.iterate (Function.swap dslope z₀) p.order f x) s
  -/
  simp only [coeff_iterate_fslope] at hx1
  /-
    case h.e'_3.h.e'_6
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    p : FormalMultilinearSeries 𝕜 𝕜 E
    f : 𝕜 → E
    z₀ : 𝕜
    hp : HasFPowerSeriesAt f p z₀
    hq : Filter.Eventually (fun z => HasSum (fun n => HSMul.hSMul (HPow.hPow (HSub …
    x : 𝕜
    hx2 : HasSum (fun n => HSMul.hSMul (HPow.hPow (HSub.hSub x z₀) n) (p.coeff n)) …
    this : ∀ (k : Nat), LT.lt k p.order → Eq (p.coeff k) 0
    s : E
    hs1 : Eq (HSMul.hSMul (HPow.hPow (HSub.hSub x z₀) p.order) s) (f x)
    hs2 : HasSum (fun m => HSMul.hSMul (HPow.hPow (HSub.hSub x z₀) m) (p.coeff (HA …
    hx1 : HasSum (fun n => HSMul.hSMul (HPow.hPow (HSub.hSub x z₀) n) (p.coeff (HA …
    ⊢ Eq (Nat.iterate (Function.swap dslope z₀) p.order f x) s
  -/
  exact hx1.unique hs2
  /-
    🎉 no goals
  -/


theorem locally_ne_zero (hp : HasFPowerSeriesAt f p z₀) (h : p ≠ 0) : ∀ᶠ z in 𝓝[≠] z₀, f z ≠ 0 := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    p : FormalMultilinearSeries 𝕜 𝕜 E
    f : 𝕜 → E
    z₀ : 𝕜
    hp : HasFPowerSeriesAt f p z₀
    h : Ne p 0
    ⊢ Filter.Eventually (fun z => Ne (f z) 0) (nhdsWithin z₀ (HasCompl.compl (Sing …
  -/
  rw [eventually_nhdsWithin_iff]
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    p : FormalMultilinearSeries 𝕜 𝕜 E
    f : 𝕜 → E
    z₀ : 𝕜
    hp : HasFPowerSeriesAt f p z₀
    h : Ne p 0
    ⊢ Filter.Eventually (fun x => Membership.mem (HasCompl.compl (Singleton.single …
  -/
  have h2 := (has_fpower_series_iterate_dslope_fslope p.order hp).continuousAt
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    p : FormalMultilinearSeries 𝕜 𝕜 E
    f : 𝕜 → E
    z₀ : 𝕜
    hp : HasFPowerSeriesAt f p z₀
    h : Ne p 0
    h2 : ContinuousAt (Nat.iterate (Function.swap dslope z₀) p.order f) z₀
    ⊢ Filter.Eventually (fun x => Membership.mem (HasCompl.compl (Singleton.single …
  -/
  have h3 := h2.eventually_ne (iterate_dslope_fslope_ne_zero hp h)
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    p : FormalMultilinearSeries 𝕜 𝕜 E
    f : 𝕜 → E
    z₀ : 𝕜
    hp : HasFPowerSeriesAt f p z₀
    h : Ne p 0
    h2 : ContinuousAt (Nat.iterate (Function.swap dslope z₀) p.order f) z₀
    h3 : Filter.Eventually (fun z => Ne (Nat.iterate (Function.swap dslope z₀) p.o …
    ⊢ Filter.Eventually (fun x => Membership.mem (HasCompl.compl (Singleton.single …
  -/
  filter_upwards [eq_pow_order_mul_iterate_dslope hp, h3] with z e1 e2 e3
  /-
    case h
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    p : FormalMultilinearSeries 𝕜 𝕜 E
    f : 𝕜 → E
    z₀ : 𝕜
    hp : HasFPowerSeriesAt f p z₀
    h : Ne p 0
    h2 : ContinuousAt (Nat.iterate (Function.swap dslope z₀) p.order f) z₀
    h3 : Filter.Eventually (fun z => Ne (Nat.iterate (Function.swap dslope z₀) p.o …
    z : 𝕜
    e1 : Eq (f z) (HSMul.hSMul (HPow.hPow (HSub.hSub z z₀) p.order) (Nat.iterate ( …
    e2 : Ne (Nat.iterate (Function.swap dslope z₀) p.order f z) 0
    e3 : Membership.mem (HasCompl.compl (Singleton.singleton z₀)) z
    ⊢ Ne (f z) 0
  -/
  simpa [e1, e2, e3] using pow_ne_zero p.order (sub_ne_zero.mpr e3)
  /-
    🎉 no goals
  -/


theorem locally_zero_iff (hp : HasFPowerSeriesAt f p z₀) : (∀ᶠ z in 𝓝 z₀, f z = 0) ↔ p = 0 :=
                                                                                   /-
                                                                                     𝕜 : Type u_1
                                                                                     inst✝² : NontriviallyNormedField 𝕜
                                                                                     E : Type u_2
                                                                                     inst✝¹ : NormedAddCommGroup E
                                                                                     inst✝ : NormedSpace 𝕜 E
                                                                                     p : FormalMultilinearSeries 𝕜 𝕜 E
                                                                                     f : 𝕜 → E
                                                                                     z₀ : 𝕜
                                                                                     hp : HasFPowerSeriesAt f p z₀
                                                                                     h : Eq p 0
                                                                                     ⊢ HasFPowerSeriesAt f 0 z₀
                                                                                   -/
  ⟨fun hf => hp.eq_zero_of_eventually hf, fun h => eventually_eq_zero (𝕜 := 𝕜) (by rwa [h] at hp)⟩
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


/-- The *principle of isolated zeros* for an analytic function, local version: if a function is
analytic at `z₀`, then either it is identically zero in a neighborhood of `z₀`, or it does not
vanish in a punctured neighborhood of `z₀`. -/
theorem eventually_eq_zero_or_eventually_ne_zero (hf : AnalyticAt 𝕜 f z₀) :
    (∀ᶠ z in 𝓝 z₀, f z = 0) ∨ ∀ᶠ z in 𝓝[≠] z₀, f z ≠ 0 := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → E
    z₀ : 𝕜
    hf : AnalyticAt 𝕜 f z₀
    ⊢ Or (Filter.Eventually (fun z => Eq (f z) 0) (nhds z₀)) (Filter.Eventually (f …
  -/
  rcases hf with ⟨p, hp⟩
  /-
    case intro
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → E
    z₀ : 𝕜
    p : FormalMultilinearSeries 𝕜 𝕜 E
    hp : HasFPowerSeriesAt f p z₀
    ⊢ Or (Filter.Eventually (fun z => Eq (f z) 0) (nhds z₀)) (Filter.Eventually (f …
  -/
  by_cases h : p = 0
    /-
      case pos
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      z₀ : 𝕜
      p : FormalMultilinearSeries 𝕜 𝕜 E
      hp : HasFPowerSeriesAt f p z₀
      h : Eq p 0
      ⊢ Or (Filter.Eventually (fun z => Eq (f z) 0) (nhds z₀)) (Filter.Eventually (f …
    -/
  · exact Or.inl (HasFPowerSeriesAt.eventually_eq_zero (by rwa [h] at hp))
    /-
      🎉 no goals
    -/
    /-
      case neg
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      z₀ : 𝕜
      p : FormalMultilinearSeries 𝕜 𝕜 E
      hp : HasFPowerSeriesAt f p z₀
      h : Not (Eq p 0)
      ⊢ Or (Filter.Eventually (fun z => Eq (f z) 0) (nhds z₀)) (Filter.Eventually (f …
    -/
  · exact Or.inr (hp.locally_ne_zero h)
    /-
      🎉 no goals
    -/


theorem eventually_eq_or_eventually_ne (hf : AnalyticAt 𝕜 f z₀) (hg : AnalyticAt 𝕜 g z₀) :
    (∀ᶠ z in 𝓝 z₀, f z = g z) ∨ ∀ᶠ z in 𝓝[≠] z₀, f z ≠ g z := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f g : 𝕜 → E
    z₀ : 𝕜
    hf : AnalyticAt 𝕜 f z₀
    hg : AnalyticAt 𝕜 g z₀
    ⊢ Or (Filter.Eventually (fun z => Eq (f z) (g z)) (nhds z₀)) (Filter.Eventuall …
  -/
  simpa [sub_eq_zero] using (hf.sub hg).eventually_eq_zero_or_eventually_ne_zero
  /-
    🎉 no goals
  -/


theorem frequently_zero_iff_eventually_zero {f : 𝕜 → E} {w : 𝕜} (hf : AnalyticAt 𝕜 f w) :
    (∃ᶠ z in 𝓝[≠] w, f z = 0) ↔ ∀ᶠ z in 𝓝 w, f z = 0 :=
  ⟨hf.eventually_eq_zero_or_eventually_ne_zero.resolve_right, fun h =>
    (h.filter_mono nhdsWithin_le_nhds).frequently⟩


theorem frequently_eq_iff_eventually_eq (hf : AnalyticAt 𝕜 f z₀) (hg : AnalyticAt 𝕜 g z₀) :
    (∃ᶠ z in 𝓝[≠] z₀, f z = g z) ↔ ∀ᶠ z in 𝓝 z₀, f z = g z := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f g : 𝕜 → E
    z₀ : 𝕜
    hf : AnalyticAt 𝕜 f z₀
    hg : AnalyticAt 𝕜 g z₀
    ⊢ Iff (Filter.Frequently (fun z => Eq (f z) (g z)) (nhdsWithin z₀ (HasCompl.co …
  -/
  simpa [sub_eq_zero] using frequently_zero_iff_eventually_zero (hf.sub hg)
  /-
    🎉 no goals
  -/


/-- For a function `f` on `𝕜`, and `z₀ ∈ 𝕜`, there exists at most one `n` such that on a punctured
neighbourhood of `z₀` we have `f z = (z - z₀) ^ n • g z`, with `g` analytic and nonvanishing at
`z₀`. We formulate this with `n : ℤ`, and deduce the case `n : ℕ` later, for applications to
meromorphic functions. -/
lemma unique_eventuallyEq_zpow_smul_nonzero {m n : ℤ}
    (hm : ∃ g, AnalyticAt 𝕜 g z₀ ∧ g z₀ ≠ 0 ∧ ∀ᶠ z in 𝓝[≠] z₀, f z = (z - z₀) ^ m • g z)
    (hn : ∃ g, AnalyticAt 𝕜 g z₀ ∧ g z₀ ≠ 0 ∧ ∀ᶠ z in 𝓝[≠] z₀, f z = (z - z₀) ^ n • g z) :
    m = n := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → E
    z₀ : 𝕜
    m n : Int
    hm : Exists fun g => And (AnalyticAt 𝕜 g z₀) (And (Ne (g z₀) 0) (Filter.Eventu …
    hn : Exists fun g => And (AnalyticAt 𝕜 g z₀) (And (Ne (g z₀) 0) (Filter.Eventu …
    ⊢ Eq m n
  -/
  wlog h_le : n ≤ m generalizing m n
    /-
      case inr
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      z₀ : 𝕜
      m n : Int
      hm : Exists fun g => And (AnalyticAt 𝕜 g z₀) (And (Ne (g z₀) 0) (Filter.Eventu …
      hn : Exists fun g => And (AnalyticAt 𝕜 g z₀) (And (Ne (g z₀) 0) (Filter.Eventu …
      this : ∀ {m n : Int}, (Exists fun g => And (AnalyticAt 𝕜 g z₀) (And (Ne (g z₀) …
      h_le : Not (LE.le n m)
      ⊢ Eq m n
    -/
  · exact ((this hn hm) (not_le.mp h_le).le).symm
    /-
      🎉 no goals
    -/
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → E
    z₀ : 𝕜
    m n : Int
    hm : Exists fun g => And (AnalyticAt 𝕜 g z₀) (And (Ne (g z₀) 0) (Filter.Eventu …
    hn : Exists fun g => And (AnalyticAt 𝕜 g z₀) (And (Ne (g z₀) 0) (Filter.Eventu …
    h_le : LE.le n m
    ⊢ Eq m n
  -/
  let ⟨g, hg_an, _, hg_eq⟩ := hm
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → E
    z₀ : 𝕜
    m n : Int
    hm : Exists fun g => And (AnalyticAt 𝕜 g z₀) (And (Ne (g z₀) 0) (Filter.Eventu …
    hn : Exists fun g => And (AnalyticAt 𝕜 g z₀) (And (Ne (g z₀) 0) (Filter.Eventu …
    h_le : LE.le n m
    g : 𝕜 → E
    hg_an : AnalyticAt 𝕜 g z₀
    left✝ : Ne (g z₀) 0
    hg_eq : Filter.Eventually (fun z => Eq (f z) (HSMul.hSMul (HPow.hPow (HSub.hSu …
    ⊢ Eq m n
  -/
  let ⟨j, hj_an, hj_ne, hj_eq⟩ := hn
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → E
    z₀ : 𝕜
    m n : Int
    hm : Exists fun g => And (AnalyticAt 𝕜 g z₀) (And (Ne (g z₀) 0) (Filter.Eventu …
    hn : Exists fun g => And (AnalyticAt 𝕜 g z₀) (And (Ne (g z₀) 0) (Filter.Eventu …
    h_le : LE.le n m
    g : 𝕜 → E
    hg_an : AnalyticAt 𝕜 g z₀
    left✝ : Ne (g z₀) 0
    hg_eq : Filter.Eventually (fun z => Eq (f z) (HSMul.hSMul (HPow.hPow (HSub.hSu …
    j : 𝕜 → E
    hj_an : AnalyticAt 𝕜 j z₀
    hj_ne : Ne (j z₀) 0
    hj_eq : Filter.Eventually (fun z => Eq (f z) (HSMul.hSMul (HPow.hPow (HSub.hSu …
    ⊢ Eq m n
  -/
  contrapose! hj_ne
  have : ∃ᶠ z in 𝓝[≠] z₀, j z = (z - z₀) ^ (m - n) • g z := by
    apply Filter.Eventually.frequently
    rw [eventually_nhdsWithin_iff] at hg_eq hj_eq ⊢
    filter_upwards [hg_eq, hj_eq] with z hfz hfz' hz
    rw [← add_sub_cancel_left n m, add_sub_assoc, zpow_add₀ <| sub_ne_zero.mpr hz, mul_smul,
      hfz' hz, smul_right_inj <| zpow_ne_zero _ <| sub_ne_zero.mpr hz] at hfz
    exact hfz hz
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → E
    z₀ : 𝕜
    m n : Int
    hm : Exists fun g => And (AnalyticAt 𝕜 g z₀) (And (Ne (g z₀) 0) (Filter.Eventu …
    hn : Exists fun g => And (AnalyticAt 𝕜 g z₀) (And (Ne (g z₀) 0) (Filter.Eventu …
    h_le : LE.le n m
    g : 𝕜 → E
    hg_an : AnalyticAt 𝕜 g z₀
    left✝ : Ne (g z₀) 0
    hg_eq : Filter.Eventually (fun z => Eq (f z) (HSMul.hSMul (HPow.hPow (HSub.hSu …
    j : 𝕜 → E
    hj_an : AnalyticAt 𝕜 j z₀
    hj_eq : Filter.Eventually (fun z => Eq (f z) (HSMul.hSMul (HPow.hPow (HSub.hSu …
    hj_ne : Ne m n
    this : Filter.Frequently (fun z => Eq (j z) (HSMul.hSMul (HPow.hPow (HSub.hSub …
    ⊢ Eq (j z₀) 0
  -/
  rw [frequently_eq_iff_eventually_eq hj_an] at this
    /-
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      z₀ : 𝕜
      m n : Int
      hm : Exists fun g => And (AnalyticAt 𝕜 g z₀) (And (Ne (g z₀) 0) (Filter.Eventu …
      hn : Exists fun g => And (AnalyticAt 𝕜 g z₀) (And (Ne (g z₀) 0) (Filter.Eventu …
      h_le : LE.le n m
      g : 𝕜 → E
      hg_an : AnalyticAt 𝕜 g z₀
      left✝ : Ne (g z₀) 0
      hg_eq : Filter.Eventually (fun z => Eq (f z) (HSMul.hSMul (HPow.hPow (HSub.hSu …
      j : 𝕜 → E
      hj_an : AnalyticAt 𝕜 j z₀
      hj_eq : Filter.Eventually (fun z => Eq (f z) (HSMul.hSMul (HPow.hPow (HSub.hSu …
      hj_ne : Ne m n
      this : Filter.Eventually (fun z => Eq (j z) (HSMul.hSMul (HPow.hPow (HSub.hSub …
      ⊢ Eq (j z₀) 0
    -/
  · rw [EventuallyEq.eq_of_nhds this, sub_self, zero_zpow _ (sub_ne_zero.mpr hj_ne), zero_smul]
    /-
      🎉 no goals
    -/
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → E
    z₀ : 𝕜
    m n : Int
    hm : Exists fun g => And (AnalyticAt 𝕜 g z₀) (And (Ne (g z₀) 0) (Filter.Eventu …
    hn : Exists fun g => And (AnalyticAt 𝕜 g z₀) (And (Ne (g z₀) 0) (Filter.Eventu …
    h_le : LE.le n m
    g : 𝕜 → E
    hg_an : AnalyticAt 𝕜 g z₀
    left✝ : Ne (g z₀) 0
    hg_eq : Filter.Eventually (fun z => Eq (f z) (HSMul.hSMul (HPow.hPow (HSub.hSu …
    j : 𝕜 → E
    hj_an : AnalyticAt 𝕜 j z₀
    hj_eq : Filter.Eventually (fun z => Eq (f z) (HSMul.hSMul (HPow.hPow (HSub.hSu …
    hj_ne : Ne m n
    this : Filter.Frequently (fun z => Eq (j z) (HSMul.hSMul (HPow.hPow (HSub.hSub …
    ⊢ AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z z₀) (HSub.hSub m  …
  -/
  conv => enter [2, z, 1]; rw [← Int.toNat_sub_of_le h_le, zpow_natCast]
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → E
    z₀ : 𝕜
    m n : Int
    hm : Exists fun g => And (AnalyticAt 𝕜 g z₀) (And (Ne (g z₀) 0) (Filter.Eventu …
    hn : Exists fun g => And (AnalyticAt 𝕜 g z₀) (And (Ne (g z₀) 0) (Filter.Eventu …
    h_le : LE.le n m
    g : 𝕜 → E
    hg_an : AnalyticAt 𝕜 g z₀
    left✝ : Ne (g z₀) 0
    hg_eq : Filter.Eventually (fun z => Eq (f z) (HSMul.hSMul (HPow.hPow (HSub.hSu …
    j : 𝕜 → E
    hj_an : AnalyticAt 𝕜 j z₀
    hj_eq : Filter.Eventually (fun z => Eq (f z) (HSMul.hSMul (HPow.hPow (HSub.hSu …
    hj_ne : Ne m n
    this : Filter.Frequently (fun z => Eq (j z) (HSMul.hSMul (HPow.hPow (HSub.hSub …
    ⊢ AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z z₀) (HSub.hSub m  …
  -/
  exact ((analyticAt_id.sub analyticAt_const).pow _).smul hg_an
  /-
    🎉 no goals
  -/


/-- For a function `f` on `𝕜`, and `z₀ ∈ 𝕜`, there exists at most one `n` such that on a
neighbourhood of `z₀` we have `f z = (z - z₀) ^ n • g z`, with `g` analytic and nonvanishing at
`z₀`. -/
lemma unique_eventuallyEq_pow_smul_nonzero {m n : ℕ}
    (hm : ∃ g, AnalyticAt 𝕜 g z₀ ∧ g z₀ ≠ 0 ∧ ∀ᶠ z in 𝓝 z₀, f z = (z - z₀) ^ m • g z)
    (hn : ∃ g, AnalyticAt 𝕜 g z₀ ∧ g z₀ ≠ 0 ∧ ∀ᶠ z in 𝓝 z₀, f z = (z - z₀) ^ n • g z) :
    m = n := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → E
    z₀ : 𝕜
    m n : Nat
    hm : Exists fun g => And (AnalyticAt 𝕜 g z₀) (And (Ne (g z₀) 0) (Filter.Eventu …
    hn : Exists fun g => And (AnalyticAt 𝕜 g z₀) (And (Ne (g z₀) 0) (Filter.Eventu …
    ⊢ Eq m n
  -/
  simp_rw [← zpow_natCast] at hm hn
  exact Int.ofNat_inj.mp <| unique_eventuallyEq_zpow_smul_nonzero
    (let ⟨g, h₁, h₂, h₃⟩ := hm; ⟨g, h₁, h₂, h₃.filter_mono nhdsWithin_le_nhds⟩)
    (let ⟨g, h₁, h₂, h₃⟩ := hn; ⟨g, h₁, h₂, h₃.filter_mono nhdsWithin_le_nhds⟩)


/-- If `f` is analytic at `z₀`, then exactly one of the following two possibilities occurs: either
`f` vanishes identically near `z₀`, or locally around `z₀` it has the form `z ↦ (z - z₀) ^ n • g z`
for some `n` and some `g` which is analytic and non-vanishing at `z₀`. -/
theorem exists_eventuallyEq_pow_smul_nonzero_iff (hf : AnalyticAt 𝕜 f z₀) :
    (∃ (n : ℕ), ∃ (g : 𝕜 → E), AnalyticAt 𝕜 g z₀ ∧ g z₀ ≠ 0 ∧
    ∀ᶠ z in 𝓝 z₀, f z = (z - z₀) ^ n • g z) ↔ (¬∀ᶠ z in 𝓝 z₀, f z = 0) := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → E
    z₀ : 𝕜
    hf : AnalyticAt 𝕜 f z₀
    ⊢ Iff (Exists fun n => Exists fun g => And (AnalyticAt 𝕜 g z₀) (And (Ne (g z₀) …
  -/
  constructor
    /-
      case mp
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      z₀ : 𝕜
      hf : AnalyticAt 𝕜 f z₀
      ⊢ (Exists fun n => Exists fun g => And (AnalyticAt 𝕜 g z₀) (And (Ne (g z₀) 0)  …
    -/
  · rintro ⟨n, g, hg_an, hg_ne, hg_eq⟩
    /-
      case mp.intro.intro.intro.intro
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      z₀ : 𝕜
      hf : AnalyticAt 𝕜 f z₀
      n : Nat
      g : 𝕜 → E
      hg_an : AnalyticAt 𝕜 g z₀
      hg_ne : Ne (g z₀) 0
      hg_eq : Filter.Eventually (fun z => Eq (f z) (HSMul.hSMul (HPow.hPow (HSub.hSu …
      ⊢ Not (Filter.Eventually (fun z => Eq (f z) 0) (nhds z₀))
    -/
    contrapose! hg_ne
    /-
      case mp.intro.intro.intro.intro
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      z₀ : 𝕜
      hf : AnalyticAt 𝕜 f z₀
      n : Nat
      g : 𝕜 → E
      hg_an : AnalyticAt 𝕜 g z₀
      hg_eq : Filter.Eventually (fun z => Eq (f z) (HSMul.hSMul (HPow.hPow (HSub.hSu …
      hg_ne : Filter.Eventually (fun z => Eq (f z) 0) (nhds z₀)
      ⊢ Eq (g z₀) 0
    -/
    apply EventuallyEq.eq_of_nhds
    /-
      case mp.intro.intro.intro.intro.h
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      z₀ : 𝕜
      hf : AnalyticAt 𝕜 f z₀
      n : Nat
      g : 𝕜 → E
      hg_an : AnalyticAt 𝕜 g z₀
      hg_eq : Filter.Eventually (fun z => Eq (f z) (HSMul.hSMul (HPow.hPow (HSub.hSu …
      hg_ne : Filter.Eventually (fun z => Eq (f z) 0) (nhds z₀)
      ⊢ (nhds z₀).EventuallyEq g fun {z₀} => 0
    -/
    rw [EventuallyEq, ← AnalyticAt.frequently_eq_iff_eventually_eq hg_an analyticAt_const]
    /-
      case mp.intro.intro.intro.intro.h
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      z₀ : 𝕜
      hf : AnalyticAt 𝕜 f z₀
      n : Nat
      g : 𝕜 → E
      hg_an : AnalyticAt 𝕜 g z₀
      hg_eq : Filter.Eventually (fun z => Eq (f z) (HSMul.hSMul (HPow.hPow (HSub.hSu …
      hg_ne : Filter.Eventually (fun z => Eq (f z) 0) (nhds z₀)
      ⊢ Filter.Frequently (fun z => Eq (g z) 0) (nhdsWithin z₀ (HasCompl.compl (Sing …
    -/
    refine (eventually_nhdsWithin_iff.mpr ?_).frequently
    /-
      case mp.intro.intro.intro.intro.h
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      z₀ : 𝕜
      hf : AnalyticAt 𝕜 f z₀
      n : Nat
      g : 𝕜 → E
      hg_an : AnalyticAt 𝕜 g z₀
      hg_eq : Filter.Eventually (fun z => Eq (f z) (HSMul.hSMul (HPow.hPow (HSub.hSu …
      hg_ne : Filter.Eventually (fun z => Eq (f z) 0) (nhds z₀)
      ⊢ Filter.Eventually (fun x => Membership.mem (HasCompl.compl (Singleton.single …
    -/
    filter_upwards [hg_eq, hg_ne] with z hf_eq hf0 hz
    /-
      case h
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      z₀ : 𝕜
      hf : AnalyticAt 𝕜 f z₀
      n : Nat
      g : 𝕜 → E
      hg_an : AnalyticAt 𝕜 g z₀
      hg_eq : Filter.Eventually (fun z => Eq (f z) (HSMul.hSMul (HPow.hPow (HSub.hSu …
      hg_ne : Filter.Eventually (fun z => Eq (f z) 0) (nhds z₀)
      z : 𝕜
      hf_eq : Eq (f z) (HSMul.hSMul (HPow.hPow (HSub.hSub z z₀) n) (g z))
      hf0 : Eq (f z) 0
      hz : Membership.mem (HasCompl.compl (Singleton.singleton z₀)) z
      ⊢ Eq (g z) 0
    -/
    rwa [hf0, eq_comm, smul_eq_zero_iff_right] at hf_eq
    /-
      case h
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      z₀ : 𝕜
      hf : AnalyticAt 𝕜 f z₀
      n : Nat
      g : 𝕜 → E
      hg_an : AnalyticAt 𝕜 g z₀
      hg_eq : Filter.Eventually (fun z => Eq (f z) (HSMul.hSMul (HPow.hPow (HSub.hSu …
      hg_ne : Filter.Eventually (fun z => Eq (f z) 0) (nhds z₀)
      z : 𝕜
      hf_eq : Eq (HSMul.hSMul (HPow.hPow (HSub.hSub z z₀) n) (g z)) 0
      hf0 : Eq (f z) 0
      hz : Membership.mem (HasCompl.compl (Singleton.singleton z₀)) z
      ⊢ Ne (HPow.hPow (HSub.hSub z z₀) n) 0
    -/
    exact pow_ne_zero _ (sub_ne_zero.mpr hz)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      z₀ : 𝕜
      hf : AnalyticAt 𝕜 f z₀
      ⊢ Not (Filter.Eventually (fun z => Eq (f z) 0) (nhds z₀)) → Exists fun n => Ex …
    -/
  · intro hf_ne
    /-
      case mpr
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      z₀ : 𝕜
      hf : AnalyticAt 𝕜 f z₀
      hf_ne : Not (Filter.Eventually (fun z => Eq (f z) 0) (nhds z₀))
      ⊢ Exists fun n => Exists fun g => And (AnalyticAt 𝕜 g z₀) (And (Ne (g z₀) 0) ( …
    -/
    rcases hf with ⟨p, hp⟩
    exact ⟨p.order, _, ⟨_, hp.has_fpower_series_iterate_dslope_fslope p.order⟩,
      hp.iterate_dslope_fslope_ne_zero (hf_ne.imp hp.locally_zero_iff.mpr),
      hp.eq_pow_order_mul_iterate_dslope⟩


open scoped Classical in
/-- The order of vanishing of `f` at `z₀`, as an element of `ℕ∞`.

This is defined to be `∞` if `f` is identically 0 on a neighbourhood of `z₀`, and otherwise the
unique `n` such that `f z = (z - z₀) ^ n • g z` with `g` analytic and non-vanishing at `z₀`. See
`AnalyticAt.order_eq_top_iff` and `AnalyticAt.order_eq_nat_iff` for these equivalences. -/
noncomputable def order (hf : AnalyticAt 𝕜 f z₀) : ENat :=
  if h : ∀ᶠ z in 𝓝 z₀, f z = 0 then ⊤
  else ↑(hf.exists_eventuallyEq_pow_smul_nonzero_iff.mpr h).choose


lemma order_eq_top_iff (hf : AnalyticAt 𝕜 f z₀) : hf.order = ⊤ ↔ ∀ᶠ z in 𝓝 z₀, f z = 0 := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → E
    z₀ : 𝕜
    hf : AnalyticAt 𝕜 f z₀
    ⊢ Iff (Eq hf.order Top.top) (Filter.Eventually (fun z => Eq (f z) 0) (nhds z₀))
  -/
  unfold order
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → E
    z₀ : 𝕜
    hf : AnalyticAt 𝕜 f z₀
    ⊢ Iff (Eq (dite (Filter.Eventually (fun z => Eq (f z) 0) (nhds z₀)) (fun h =>  …
  -/
  split_ifs with h
    /-
      case pos
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      z₀ : 𝕜
      hf : AnalyticAt 𝕜 f z₀
      h : Filter.Eventually (fun z => Eq (f z) 0) (nhds z₀)
      ⊢ Iff (Eq Top.top Top.top) (Filter.Eventually (fun z => Eq (f z) 0) (nhds z₀))
    -/
  · rwa [eq_self, true_iff]
    /-
      🎉 no goals
    -/
    /-
      case neg
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      z₀ : 𝕜
      hf : AnalyticAt 𝕜 f z₀
      h : Not (Filter.Eventually (fun z => Eq (f z) 0) (nhds z₀))
      ⊢ Iff (Eq (↑⋯.choose) Top.top) (Filter.Eventually (fun z => Eq (f z) 0) (nhds  …
    -/
  · simpa only [ne_eq, ENat.coe_ne_top, false_iff] using h
    /-
      🎉 no goals
    -/


lemma order_eq_nat_iff (hf : AnalyticAt 𝕜 f z₀) (n : ℕ) : hf.order = ↑n ↔
    ∃ (g : 𝕜 → E), AnalyticAt 𝕜 g z₀ ∧ g z₀ ≠ 0 ∧ ∀ᶠ z in 𝓝 z₀, f z = (z - z₀) ^ n • g z := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → E
    z₀ : 𝕜
    hf : AnalyticAt 𝕜 f z₀
    n : Nat
    ⊢ Iff (Eq hf.order ↑n) (Exists fun g => And (AnalyticAt 𝕜 g z₀) (And (Ne (g z₀ …
  -/
  unfold order
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → E
    z₀ : 𝕜
    hf : AnalyticAt 𝕜 f z₀
    n : Nat
    ⊢ Iff (Eq (dite (Filter.Eventually (fun z => Eq (f z) 0) (nhds z₀)) (fun h =>  …
  -/
  split_ifs with h
    /-
      case pos
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      z₀ : 𝕜
      hf : AnalyticAt 𝕜 f z₀
      n : Nat
      h : Filter.Eventually (fun z => Eq (f z) 0) (nhds z₀)
      ⊢ Iff (Eq Top.top ↑n) (Exists fun g => And (AnalyticAt 𝕜 g z₀) (And (Ne (g z₀) …
    -/
  · simp only [ENat.top_ne_coe, false_iff]
    /-
      case pos
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      z₀ : 𝕜
      hf : AnalyticAt 𝕜 f z₀
      n : Nat
      h : Filter.Eventually (fun z => Eq (f z) 0) (nhds z₀)
      ⊢ Not (Exists fun g => And (AnalyticAt 𝕜 g z₀) (And (Ne (g z₀) 0) (Filter.Even …
    -/
    contrapose! h
    /-
      case pos
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      z₀ : 𝕜
      hf : AnalyticAt 𝕜 f z₀
      n : Nat
      h : Exists fun g => And (AnalyticAt 𝕜 g z₀) (And (Ne (g z₀) 0) (Filter.Eventua …
      ⊢ Not (Filter.Eventually (fun z => Eq (f z) 0) (nhds z₀))
    -/
    rw [← hf.exists_eventuallyEq_pow_smul_nonzero_iff]
    /-
      case pos
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      z₀ : 𝕜
      hf : AnalyticAt 𝕜 f z₀
      n : Nat
      h : Exists fun g => And (AnalyticAt 𝕜 g z₀) (And (Ne (g z₀) 0) (Filter.Eventua …
      ⊢ Exists fun n => Exists fun g => And (AnalyticAt 𝕜 g z₀) (And (Ne (g z₀) 0) ( …
    -/
    exact ⟨n, h⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      z₀ : 𝕜
      hf : AnalyticAt 𝕜 f z₀
      n : Nat
      h : Not (Filter.Eventually (fun z => Eq (f z) 0) (nhds z₀))
      ⊢ Iff (Eq ↑⋯.choose ↑n) (Exists fun g => And (AnalyticAt 𝕜 g z₀) (And (Ne (g z …
    -/
  · rw [← hf.exists_eventuallyEq_pow_smul_nonzero_iff] at h
    /-
      case neg
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      z₀ : 𝕜
      hf : AnalyticAt 𝕜 f z₀
      n : Nat
      h✝ : Not (Filter.Eventually (fun z => Eq (f z) 0) (nhds z₀))
      h : Exists fun n => Exists fun g => And (AnalyticAt 𝕜 g z₀) (And (Ne (g z₀) 0) …
      ⊢ Iff (Eq ↑⋯.choose ↑n) (Exists fun g => And (AnalyticAt 𝕜 g z₀) (And (Ne (g z …
    -/
    refine ⟨fun hn ↦ (WithTop.coe_inj.mp hn : h.choose = n) ▸ h.choose_spec, fun h' ↦ ?_⟩
    /-
      case neg
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      z₀ : 𝕜
      hf : AnalyticAt 𝕜 f z₀
      n : Nat
      h✝ : Not (Filter.Eventually (fun z => Eq (f z) 0) (nhds z₀))
      h : Exists fun n => Exists fun g => And (AnalyticAt 𝕜 g z₀) (And (Ne (g z₀) 0) …
      h' : Exists fun g => And (AnalyticAt 𝕜 g z₀) (And (Ne (g z₀) 0) (Filter.Eventu …
      ⊢ Eq ↑⋯.choose ↑n
    -/
    rw [unique_eventuallyEq_pow_smul_nonzero h.choose_spec h']
    /-
      🎉 no goals
    -/


/-- The *principle of isolated zeros* for an analytic function, global version: if a function is
analytic on a connected set `U` and vanishes in arbitrary neighborhoods of a point `z₀ ∈ U`, then
it is identically zero in `U`.
For higher-dimensional versions requiring that the function vanishes in a neighborhood of `z₀`,
see `AnalyticOnNhd.eqOn_zero_of_preconnected_of_eventuallyEq_zero`. -/
theorem eqOn_zero_of_preconnected_of_frequently_eq_zero (hf : AnalyticOnNhd 𝕜 f U)
    (hU : IsPreconnected U) (h₀ : z₀ ∈ U) (hfw : ∃ᶠ z in 𝓝[≠] z₀, f z = 0) : EqOn f 0 U :=
  hf.eqOn_zero_of_preconnected_of_eventuallyEq_zero hU h₀
    ((hf z₀ h₀).frequently_zero_iff_eventually_zero.1 hfw)


theorem eqOn_zero_or_eventually_ne_zero_of_preconnected (hf : AnalyticOnNhd 𝕜 f U)
    (hU : IsPreconnected U) : EqOn f 0 U ∨ ∀ᶠ x in codiscreteWithin U, f x ≠ 0 := by
  simp only [or_iff_not_imp_right, ne_eq, eventually_iff, mem_codiscreteWithin,
    disjoint_principal_right, not_forall]
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → E
    U : Set 𝕜
    hf : AnalyticOnNhd 𝕜 f U
    hU : IsPreconnected U
    ⊢ (Exists fun x => Exists fun x_1 => Not (Membership.mem (nhdsWithin x (HasCom …
  -/
  rintro ⟨x, hx, hx2⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → E
    U : Set 𝕜
    hf : AnalyticOnNhd 𝕜 f U
    hU : IsPreconnected U
    x : 𝕜
    hx : Membership.mem U x
    hx2 : Not (Membership.mem (nhdsWithin x (HasCompl.compl (Singleton.singleton x …
    ⊢ Set.EqOn f 0 U
  -/
  refine hf.eqOn_zero_of_preconnected_of_frequently_eq_zero hU hx fun nh ↦ hx2 ?_
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → E
    U : Set 𝕜
    hf : AnalyticOnNhd 𝕜 f U
    hU : IsPreconnected U
    x : 𝕜
    hx : Membership.mem U x
    hx2 : Not (Membership.mem (nhdsWithin x (HasCompl.compl (Singleton.singleton x …
    nh : Filter.Eventually (fun x => Not ((fun z => Eq (f z) 0) x)) (nhdsWithin x  …
    ⊢ Membership.mem (nhdsWithin x (HasCompl.compl (Singleton.singleton x))) (HasC …
  -/
  filter_upwards [nh] with a ha
  /-
    case h
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → E
    U : Set 𝕜
    hf : AnalyticOnNhd 𝕜 f U
    hU : IsPreconnected U
    x : 𝕜
    hx : Membership.mem U x
    hx2 : Not (Membership.mem (nhdsWithin x (HasCompl.compl (Singleton.singleton x …
    nh : Filter.Eventually (fun x => Not ((fun z => Eq (f z) 0) x)) (nhdsWithin x  …
    a : 𝕜
    ha : Not (Eq (f a) 0)
    ⊢ Membership.mem (HasCompl.compl (SDiff.sdiff U (setOf fun x => Not (Eq (f x)  …
  -/
  simp_all
  /-
    🎉 no goals
  -/


theorem eqOn_zero_of_preconnected_of_mem_closure (hf : AnalyticOnNhd 𝕜 f U) (hU : IsPreconnected U)
    (h₀ : z₀ ∈ U) (hfz₀ : z₀ ∈ closure ({z | f z = 0} \ {z₀})) : EqOn f 0 U :=
  hf.eqOn_zero_of_preconnected_of_frequently_eq_zero hU h₀
    (mem_closure_ne_iff_frequently_within.mp hfz₀)


/-- The *identity principle* for analytic functions, global version: if two functions are
analytic on a connected set `U` and coincide at points which accumulate to a point `z₀ ∈ U`, then
they coincide globally in `U`.
For higher-dimensional versions requiring that the functions coincide in a neighborhood of `z₀`,
see `AnalyticOnNhd.eqOn_of_preconnected_of_eventuallyEq`. -/
theorem eqOn_of_preconnected_of_frequently_eq (hf : AnalyticOnNhd 𝕜 f U) (hg : AnalyticOnNhd 𝕜 g U)
    (hU : IsPreconnected U) (h₀ : z₀ ∈ U) (hfg : ∃ᶠ z in 𝓝[≠] z₀, f z = g z) : EqOn f g U := by
  have hfg' : ∃ᶠ z in 𝓝[≠] z₀, (f - g) z = 0 :=
    hfg.mono fun z h => by rw [Pi.sub_apply, h, sub_self]
  simpa [sub_eq_zero] using fun z hz =>
    (hf.sub hg).eqOn_zero_of_preconnected_of_frequently_eq_zero hU h₀ hfg' hz


theorem eqOn_or_eventually_ne_of_preconnected (hf : AnalyticOnNhd 𝕜 f U) (hg : AnalyticOnNhd 𝕜 g U)
    (hU : IsPreconnected U) : EqOn f g U ∨ ∀ᶠ x in codiscreteWithin U, f x ≠ g x :=
  (eqOn_zero_or_eventually_ne_zero_of_preconnected (hf.sub hg) hU).imp
    (fun h _ hx ↦ eq_of_sub_eq_zero (h hx))
        /-
          𝕜 : Type u_1
          inst✝² : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝¹ : NormedAddCommGroup E
          inst✝ : NormedSpace 𝕜 E
          f g : 𝕜 → E
          U : Set 𝕜
          hf : AnalyticOnNhd 𝕜 f U
          hg : AnalyticOnNhd 𝕜 g U
          hU : IsPreconnected U
          ⊢ Filter.Eventually (fun x => Ne (HSub.hSub f g x) 0) (Filter.codiscreteWithin …
        -/
    (by simp only [Pi.sub_apply, ne_eq, sub_eq_zero, imp_self])
        /-
          🎉 no goals
        -/


theorem eqOn_of_preconnected_of_mem_closure (hf : AnalyticOnNhd 𝕜 f U) (hg : AnalyticOnNhd 𝕜 g U)
    (hU : IsPreconnected U) (h₀ : z₀ ∈ U) (hfg : z₀ ∈ closure ({z | f z = g z} \ {z₀})) :
    EqOn f g U :=
  hf.eqOn_of_preconnected_of_frequently_eq hg hU h₀ (mem_closure_ne_iff_frequently_within.mp hfg)


/-- The *identity principle* for analytic functions, global version: if two functions on a normed
field `𝕜` are analytic everywhere and coincide at points which accumulate to a point `z₀`, then
they coincide globally.
For higher-dimensional versions requiring that the functions coincide in a neighborhood of `z₀`,
see `AnalyticOnNhd.eq_of_eventuallyEq`. -/
theorem eq_of_frequently_eq [ConnectedSpace 𝕜] (hf : AnalyticOnNhd 𝕜 f univ)
    (hg : AnalyticOnNhd 𝕜 g univ) (hfg : ∃ᶠ z in 𝓝[≠] z₀, f z = g z) : f = g :=
  funext fun x =>
    eqOn_of_preconnected_of_frequently_eq hf hg isPreconnected_univ (mem_univ z₀) hfg (mem_univ x)


@[deprecated (since := "2024-09-26")]
alias _root_.AnalyticOn.eq_of_frequently_eq := eq_of_frequently_eq


