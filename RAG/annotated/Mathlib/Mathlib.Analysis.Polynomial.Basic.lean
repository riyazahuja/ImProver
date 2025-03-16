theorem eventually_no_roots (hP : P ≠ 0) : ∀ᶠ x in atTop, ¬P.IsRoot x :=
  atTop_le_cofinite <| (finite_setOf_isRoot hP).compl_mem_cofinite


theorem isEquivalent_atTop_lead :
    (fun x => eval x P) ~[atTop] fun x => P.leadingCoeff * x ^ P.natDegree := by
  /-
    𝕜 : Type u_1
    inst✝¹ : NormedLinearOrderedField 𝕜
    P : Polynomial 𝕜
    inst✝ : OrderTopology 𝕜
    ⊢ Asymptotics.IsEquivalent Filter.atTop (fun x => Polynomial.eval x P) fun x = …
  -/
  by_cases h : P = 0
    /-
      case pos
      𝕜 : Type u_1
      inst✝¹ : NormedLinearOrderedField 𝕜
      P : Polynomial 𝕜
      inst✝ : OrderTopology 𝕜
      h : Eq P 0
      ⊢ Asymptotics.IsEquivalent Filter.atTop (fun x => Polynomial.eval x P) fun x = …
    -/
  · simp [h, IsEquivalent.refl]
    /-
      🎉 no goals
    -/
    /-
      case neg
      𝕜 : Type u_1
      inst✝¹ : NormedLinearOrderedField 𝕜
      P : Polynomial 𝕜
      inst✝ : OrderTopology 𝕜
      h : Not (Eq P 0)
      ⊢ Asymptotics.IsEquivalent Filter.atTop (fun x => Polynomial.eval x P) fun x = …
    -/
  · simp only [Polynomial.eval_eq_sum_range, sum_range_succ]
    exact
      IsLittleO.add_isEquivalent
        (IsLittleO.sum fun i hi =>
          IsLittleO.const_mul_left
            ((IsLittleO.const_mul_right fun hz => h <| leadingCoeff_eq_zero.mp hz) <|
              isLittleO_pow_pow_atTop_of_lt (mem_range.mp hi))
            _)
        IsEquivalent.refl


theorem tendsto_atTop_of_leadingCoeff_nonneg (hdeg : 0 < P.degree) (hnng : 0 ≤ P.leadingCoeff) :
    Tendsto (fun x => eval x P) atTop atTop :=
  P.isEquivalent_atTop_lead.symm.tendsto_atTop <|
    tendsto_const_mul_pow_atTop (natDegree_pos_iff_degree_pos.2 hdeg).ne' <|
      hnng.lt_of_ne' <| leadingCoeff_ne_zero.mpr <| ne_zero_of_degree_gt hdeg


theorem tendsto_atTop_iff_leadingCoeff_nonneg :
    Tendsto (fun x => eval x P) atTop atTop ↔ 0 < P.degree ∧ 0 ≤ P.leadingCoeff := by
  /-
    𝕜 : Type u_1
    inst✝¹ : NormedLinearOrderedField 𝕜
    P : Polynomial 𝕜
    inst✝ : OrderTopology 𝕜
    ⊢ Iff (Filter.Tendsto (fun x => Polynomial.eval x P) Filter.atTop Filter.atTop …
  -/
  refine ⟨fun h => ?_, fun h => tendsto_atTop_of_leadingCoeff_nonneg P h.1 h.2⟩
  have : Tendsto (fun x => P.leadingCoeff * x ^ P.natDegree) atTop atTop :=
    (isEquivalent_atTop_lead P).tendsto_atTop h
  /-
    𝕜 : Type u_1
    inst✝¹ : NormedLinearOrderedField 𝕜
    P : Polynomial 𝕜
    inst✝ : OrderTopology 𝕜
    h : Filter.Tendsto (fun x => Polynomial.eval x P) Filter.atTop Filter.atTop
    this : Filter.Tendsto (fun x => HMul.hMul P.leadingCoeff (HPow.hPow x P.natDeg …
    ⊢ And (LT.lt 0 P.degree) (LE.le 0 P.leadingCoeff)
  -/
  rw [tendsto_const_mul_pow_atTop_iff, ← pos_iff_ne_zero, natDegree_pos_iff_degree_pos] at this
  /-
    𝕜 : Type u_1
    inst✝¹ : NormedLinearOrderedField 𝕜
    P : Polynomial 𝕜
    inst✝ : OrderTopology 𝕜
    h : Filter.Tendsto (fun x => Polynomial.eval x P) Filter.atTop Filter.atTop
    this : And (LT.lt 0 P.degree) (LT.lt 0 P.leadingCoeff)
    ⊢ And (LT.lt 0 P.degree) (LE.le 0 P.leadingCoeff)
  -/
  exact ⟨this.1, this.2.le⟩
  /-
    🎉 no goals
  -/


theorem tendsto_atBot_iff_leadingCoeff_nonpos :
    Tendsto (fun x => eval x P) atTop atBot ↔ 0 < P.degree ∧ P.leadingCoeff ≤ 0 := by
  simp only [← tendsto_neg_atTop_iff, ← eval_neg, tendsto_atTop_iff_leadingCoeff_nonneg,
    degree_neg, leadingCoeff_neg, neg_nonneg]


theorem tendsto_atBot_of_leadingCoeff_nonpos (hdeg : 0 < P.degree) (hnps : P.leadingCoeff ≤ 0) :
    Tendsto (fun x => eval x P) atTop atBot :=
  P.tendsto_atBot_iff_leadingCoeff_nonpos.2 ⟨hdeg, hnps⟩


theorem abs_tendsto_atTop (hdeg : 0 < P.degree) :
    Tendsto (fun x => abs <| eval x P) atTop atTop := by
  /-
    𝕜 : Type u_1
    inst✝¹ : NormedLinearOrderedField 𝕜
    P : Polynomial 𝕜
    inst✝ : OrderTopology 𝕜
    hdeg : LT.lt 0 P.degree
    ⊢ Filter.Tendsto (fun x => abs (Polynomial.eval x P)) Filter.atTop Filter.atTop
  -/
  rcases le_total 0 P.leadingCoeff with hP | hP
    /-
      case inl
      𝕜 : Type u_1
      inst✝¹ : NormedLinearOrderedField 𝕜
      P : Polynomial 𝕜
      inst✝ : OrderTopology 𝕜
      hdeg : LT.lt 0 P.degree
      hP : LE.le 0 P.leadingCoeff
      ⊢ Filter.Tendsto (fun x => abs (Polynomial.eval x P)) Filter.atTop Filter.atTop
    -/
  · exact tendsto_abs_atTop_atTop.comp (P.tendsto_atTop_of_leadingCoeff_nonneg hdeg hP)
    /-
      🎉 no goals
    -/
    /-
      case inr
      𝕜 : Type u_1
      inst✝¹ : NormedLinearOrderedField 𝕜
      P : Polynomial 𝕜
      inst✝ : OrderTopology 𝕜
      hdeg : LT.lt 0 P.degree
      hP : LE.le P.leadingCoeff 0
      ⊢ Filter.Tendsto (fun x => abs (Polynomial.eval x P)) Filter.atTop Filter.atTop
    -/
  · exact tendsto_abs_atBot_atTop.comp (P.tendsto_atBot_of_leadingCoeff_nonpos hdeg hP)
    /-
      🎉 no goals
    -/


theorem abs_isBoundedUnder_iff :
    (IsBoundedUnder (· ≤ ·) atTop fun x => |eval x P|) ↔ P.degree ≤ 0 := by
  refine ⟨fun h => ?_, fun h => ⟨|P.coeff 0|, eventually_map.mpr (Eventually.of_forall
    (forall_imp (fun _ => le_of_eq) fun x => congr_arg abs <| _root_.trans (congr_arg (eval x)
    (eq_C_of_degree_le_zero h)) eval_C))⟩⟩
  /-
    𝕜 : Type u_1
    inst✝¹ : NormedLinearOrderedField 𝕜
    P : Polynomial 𝕜
    inst✝ : OrderTopology 𝕜
    h : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) Filter.atTop fun x => abs …
    ⊢ LE.le P.degree 0
  -/
  contrapose! h
  /-
    𝕜 : Type u_1
    inst✝¹ : NormedLinearOrderedField 𝕜
    P : Polynomial 𝕜
    inst✝ : OrderTopology 𝕜
    h : LT.lt 0 P.degree
    ⊢ Not (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) Filter.atTop fun x =>  …
  -/
  exact not_isBoundedUnder_of_tendsto_atTop (abs_tendsto_atTop P h)
  /-
    🎉 no goals
  -/


theorem abs_tendsto_atTop_iff : Tendsto (fun x => abs <| eval x P) atTop atTop ↔ 0 < P.degree :=
  ⟨fun h => not_le.mp (mt (abs_isBoundedUnder_iff P).mpr (not_isBoundedUnder_of_tendsto_atTop h)),
    abs_tendsto_atTop P⟩


theorem tendsto_nhds_iff {c : 𝕜} :
    Tendsto (fun x => eval x P) atTop (𝓝 c) ↔ P.leadingCoeff = c ∧ P.degree ≤ 0 := by
  /-
    𝕜 : Type u_1
    inst✝¹ : NormedLinearOrderedField 𝕜
    P : Polynomial 𝕜
    inst✝ : OrderTopology 𝕜
    c : 𝕜
    ⊢ Iff (Filter.Tendsto (fun x => Polynomial.eval x P) Filter.atTop (nhds c)) (A …
  -/
  refine ⟨fun h => ?_, fun h => ?_⟩
    /-
      case refine_1
      𝕜 : Type u_1
      inst✝¹ : NormedLinearOrderedField 𝕜
      P : Polynomial 𝕜
      inst✝ : OrderTopology 𝕜
      c : 𝕜
      h : Filter.Tendsto (fun x => Polynomial.eval x P) Filter.atTop (nhds c)
      ⊢ And (Eq P.leadingCoeff c) (LE.le P.degree 0)
    -/
  · have := P.isEquivalent_atTop_lead.tendsto_nhds h
    /-
      case refine_1
      𝕜 : Type u_1
      inst✝¹ : NormedLinearOrderedField 𝕜
      P : Polynomial 𝕜
      inst✝ : OrderTopology 𝕜
      c : 𝕜
      h : Filter.Tendsto (fun x => Polynomial.eval x P) Filter.atTop (nhds c)
      this : Filter.Tendsto (fun x => HMul.hMul P.leadingCoeff (HPow.hPow x P.natDeg …
      ⊢ And (Eq P.leadingCoeff c) (LE.le P.degree 0)
    -/
    by_cases hP : P.leadingCoeff = 0
      /-
        case pos
        𝕜 : Type u_1
        inst✝¹ : NormedLinearOrderedField 𝕜
        P : Polynomial 𝕜
        inst✝ : OrderTopology 𝕜
        c : 𝕜
        h : Filter.Tendsto (fun x => Polynomial.eval x P) Filter.atTop (nhds c)
        this : Filter.Tendsto (fun x => HMul.hMul P.leadingCoeff (HPow.hPow x P.natDeg …
        hP : Eq P.leadingCoeff 0
        ⊢ And (Eq P.leadingCoeff c) (LE.le P.degree 0)
      -/
    · simp only [hP, zero_mul, tendsto_const_nhds_iff] at this
      /-
        case pos
        𝕜 : Type u_1
        inst✝¹ : NormedLinearOrderedField 𝕜
        P : Polynomial 𝕜
        inst✝ : OrderTopology 𝕜
        c : 𝕜
        h : Filter.Tendsto (fun x => Polynomial.eval x P) Filter.atTop (nhds c)
        hP : Eq P.leadingCoeff 0
        this : Eq 0 c
        ⊢ And (Eq P.leadingCoeff c) (LE.le P.degree 0)
      -/
      exact ⟨_root_.trans hP this, by simp [leadingCoeff_eq_zero.1 hP]⟩
      /-
        🎉 no goals
      -/
      /-
        case neg
        𝕜 : Type u_1
        inst✝¹ : NormedLinearOrderedField 𝕜
        P : Polynomial 𝕜
        inst✝ : OrderTopology 𝕜
        c : 𝕜
        h : Filter.Tendsto (fun x => Polynomial.eval x P) Filter.atTop (nhds c)
        this : Filter.Tendsto (fun x => HMul.hMul P.leadingCoeff (HPow.hPow x P.natDeg …
        hP : Not (Eq P.leadingCoeff 0)
        ⊢ And (Eq P.leadingCoeff c) (LE.le P.degree 0)
      -/
    · rw [tendsto_const_mul_pow_nhds_iff hP, natDegree_eq_zero_iff_degree_le_zero] at this
      /-
        case neg
        𝕜 : Type u_1
        inst✝¹ : NormedLinearOrderedField 𝕜
        P : Polynomial 𝕜
        inst✝ : OrderTopology 𝕜
        c : 𝕜
        h : Filter.Tendsto (fun x => Polynomial.eval x P) Filter.atTop (nhds c)
        this : And (LE.le P.degree 0) (Eq P.leadingCoeff c)
        hP : Not (Eq P.leadingCoeff 0)
        ⊢ And (Eq P.leadingCoeff c) (LE.le P.degree 0)
      -/
      exact this.symm
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      𝕜 : Type u_1
      inst✝¹ : NormedLinearOrderedField 𝕜
      P : Polynomial 𝕜
      inst✝ : OrderTopology 𝕜
      c : 𝕜
      h : And (Eq P.leadingCoeff c) (LE.le P.degree 0)
      ⊢ Filter.Tendsto (fun x => Polynomial.eval x P) Filter.atTop (nhds c)
    -/
  · refine P.isEquivalent_atTop_lead.symm.tendsto_nhds ?_
    /-
      case refine_2
      𝕜 : Type u_1
      inst✝¹ : NormedLinearOrderedField 𝕜
      P : Polynomial 𝕜
      inst✝ : OrderTopology 𝕜
      c : 𝕜
      h : And (Eq P.leadingCoeff c) (LE.le P.degree 0)
      ⊢ Filter.Tendsto (fun x => HMul.hMul P.leadingCoeff (HPow.hPow x P.natDegree)) …
    -/
    have : P.natDegree = 0 := natDegree_eq_zero_iff_degree_le_zero.2 h.2
    /-
      case refine_2
      𝕜 : Type u_1
      inst✝¹ : NormedLinearOrderedField 𝕜
      P : Polynomial 𝕜
      inst✝ : OrderTopology 𝕜
      c : 𝕜
      h : And (Eq P.leadingCoeff c) (LE.le P.degree 0)
      this : Eq P.natDegree 0
      ⊢ Filter.Tendsto (fun x => HMul.hMul P.leadingCoeff (HPow.hPow x P.natDegree)) …
    -/
    simp only [h.1, this, pow_zero, mul_one]
    /-
      case refine_2
      𝕜 : Type u_1
      inst✝¹ : NormedLinearOrderedField 𝕜
      P : Polynomial 𝕜
      inst✝ : OrderTopology 𝕜
      c : 𝕜
      h : And (Eq P.leadingCoeff c) (LE.le P.degree 0)
      this : Eq P.natDegree 0
      ⊢ Filter.Tendsto (fun x => c) Filter.atTop (nhds c)
    -/
    exact tendsto_const_nhds
    /-
      🎉 no goals
    -/


theorem isEquivalent_atTop_div :
    (fun x => eval x P / eval x Q) ~[atTop] fun x =>
      P.leadingCoeff / Q.leadingCoeff * x ^ (P.natDegree - Q.natDegree : ℤ) := by
  /-
    𝕜 : Type u_1
    inst✝¹ : NormedLinearOrderedField 𝕜
    P Q : Polynomial 𝕜
    inst✝ : OrderTopology 𝕜
    ⊢ Asymptotics.IsEquivalent Filter.atTop (fun x => HDiv.hDiv (Polynomial.eval x …
  -/
  by_cases hP : P = 0
    /-
      case pos
      𝕜 : Type u_1
      inst✝¹ : NormedLinearOrderedField 𝕜
      P Q : Polynomial 𝕜
      inst✝ : OrderTopology 𝕜
      hP : Eq P 0
      ⊢ Asymptotics.IsEquivalent Filter.atTop (fun x => HDiv.hDiv (Polynomial.eval x …
    -/
  · simp [hP, IsEquivalent.refl]
    /-
      🎉 no goals
    -/
  /-
    case neg
    𝕜 : Type u_1
    inst✝¹ : NormedLinearOrderedField 𝕜
    P Q : Polynomial 𝕜
    inst✝ : OrderTopology 𝕜
    hP : Not (Eq P 0)
    ⊢ Asymptotics.IsEquivalent Filter.atTop (fun x => HDiv.hDiv (Polynomial.eval x …
  -/
  by_cases hQ : Q = 0
    /-
      case pos
      𝕜 : Type u_1
      inst✝¹ : NormedLinearOrderedField 𝕜
      P Q : Polynomial 𝕜
      inst✝ : OrderTopology 𝕜
      hP : Not (Eq P 0)
      hQ : Eq Q 0
      ⊢ Asymptotics.IsEquivalent Filter.atTop (fun x => HDiv.hDiv (Polynomial.eval x …
    -/
  · simp [hQ, IsEquivalent.refl]
    /-
      🎉 no goals
    -/
  refine
    (P.isEquivalent_atTop_lead.symm.div Q.isEquivalent_atTop_lead.symm).symm.trans
      (EventuallyEq.isEquivalent ((eventually_gt_atTop 0).mono fun x hx => ?_))
  /-
    case neg
    𝕜 : Type u_1
    inst✝¹ : NormedLinearOrderedField 𝕜
    P Q : Polynomial 𝕜
    inst✝ : OrderTopology 𝕜
    hP : Not (Eq P 0)
    hQ : Not (Eq Q 0)
    x : 𝕜
    hx : LT.lt 0 x
    ⊢ Eq ((fun x => HDiv.hDiv (HMul.hMul P.leadingCoeff (HPow.hPow x P.natDegree)) …
  -/
  simp [← div_mul_div_comm, hP, hQ, zpow_sub₀ hx.ne.symm]
  /-
    🎉 no goals
  -/


theorem div_tendsto_zero_of_degree_lt (hdeg : P.degree < Q.degree) :
    Tendsto (fun x => eval x P / eval x Q) atTop (𝓝 0) := by
  /-
    𝕜 : Type u_1
    inst✝¹ : NormedLinearOrderedField 𝕜
    P Q : Polynomial 𝕜
    inst✝ : OrderTopology 𝕜
    hdeg : LT.lt P.degree Q.degree
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (Polynomial.eval x P) (Polynomial.eval x  …
  -/
  by_cases hP : P = 0
    /-
      case pos
      𝕜 : Type u_1
      inst✝¹ : NormedLinearOrderedField 𝕜
      P Q : Polynomial 𝕜
      inst✝ : OrderTopology 𝕜
      hdeg : LT.lt P.degree Q.degree
      hP : Eq P 0
      ⊢ Filter.Tendsto (fun x => HDiv.hDiv (Polynomial.eval x P) (Polynomial.eval x  …
    -/
  · simp [hP, tendsto_const_nhds]
    /-
      🎉 no goals
    -/
  /-
    case neg
    𝕜 : Type u_1
    inst✝¹ : NormedLinearOrderedField 𝕜
    P Q : Polynomial 𝕜
    inst✝ : OrderTopology 𝕜
    hdeg : LT.lt P.degree Q.degree
    hP : Not (Eq P 0)
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (Polynomial.eval x P) (Polynomial.eval x  …
  -/
  rw [← natDegree_lt_natDegree_iff hP] at hdeg
  /-
    case neg
    𝕜 : Type u_1
    inst✝¹ : NormedLinearOrderedField 𝕜
    P Q : Polynomial 𝕜
    inst✝ : OrderTopology 𝕜
    hdeg : LT.lt P.natDegree Q.natDegree
    hP : Not (Eq P 0)
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (Polynomial.eval x P) (Polynomial.eval x  …
  -/
  refine (isEquivalent_atTop_div P Q).symm.tendsto_nhds ?_
  /-
    case neg
    𝕜 : Type u_1
    inst✝¹ : NormedLinearOrderedField 𝕜
    P Q : Polynomial 𝕜
    inst✝ : OrderTopology 𝕜
    hdeg : LT.lt P.natDegree Q.natDegree
    hP : Not (Eq P 0)
    ⊢ Filter.Tendsto (fun x => HMul.hMul (HDiv.hDiv P.leadingCoeff Q.leadingCoeff) …
  -/
  rw [← mul_zero]
  /-
    case neg
    𝕜 : Type u_1
    inst✝¹ : NormedLinearOrderedField 𝕜
    P Q : Polynomial 𝕜
    inst✝ : OrderTopology 𝕜
    hdeg : LT.lt P.natDegree Q.natDegree
    hP : Not (Eq P 0)
    ⊢ Filter.Tendsto (fun x => HMul.hMul (HDiv.hDiv P.leadingCoeff Q.leadingCoeff) …
  -/
  refine (tendsto_zpow_atTop_zero ?_).const_mul _
  /-
    case neg
    𝕜 : Type u_1
    inst✝¹ : NormedLinearOrderedField 𝕜
    P Q : Polynomial 𝕜
    inst✝ : OrderTopology 𝕜
    hdeg : LT.lt P.natDegree Q.natDegree
    hP : Not (Eq P 0)
    ⊢ LT.lt (HSub.hSub ↑P.natDegree ↑Q.natDegree) 0
  -/
  omega
  /-
    🎉 no goals
  -/


theorem div_tendsto_zero_iff_degree_lt (hQ : Q ≠ 0) :
    Tendsto (fun x => eval x P / eval x Q) atTop (𝓝 0) ↔ P.degree < Q.degree := by
  /-
    𝕜 : Type u_1
    inst✝¹ : NormedLinearOrderedField 𝕜
    P Q : Polynomial 𝕜
    inst✝ : OrderTopology 𝕜
    hQ : Ne Q 0
    ⊢ Iff (Filter.Tendsto (fun x => HDiv.hDiv (Polynomial.eval x P) (Polynomial.ev …
  -/
  refine ⟨fun h => ?_, div_tendsto_zero_of_degree_lt P Q⟩
  /-
    𝕜 : Type u_1
    inst✝¹ : NormedLinearOrderedField 𝕜
    P Q : Polynomial 𝕜
    inst✝ : OrderTopology 𝕜
    hQ : Ne Q 0
    h : Filter.Tendsto (fun x => HDiv.hDiv (Polynomial.eval x P) (Polynomial.eval  …
    ⊢ LT.lt P.degree Q.degree
  -/
  by_cases hPQ : P.leadingCoeff / Q.leadingCoeff = 0
    /-
      case pos
      𝕜 : Type u_1
      inst✝¹ : NormedLinearOrderedField 𝕜
      P Q : Polynomial 𝕜
      inst✝ : OrderTopology 𝕜
      hQ : Ne Q 0
      h : Filter.Tendsto (fun x => HDiv.hDiv (Polynomial.eval x P) (Polynomial.eval  …
      hPQ : Eq (HDiv.hDiv P.leadingCoeff Q.leadingCoeff) 0
      ⊢ LT.lt P.degree Q.degree
    -/
  · simp only [div_eq_mul_inv, inv_eq_zero, mul_eq_zero] at hPQ
    /-
      case pos
      𝕜 : Type u_1
      inst✝¹ : NormedLinearOrderedField 𝕜
      P Q : Polynomial 𝕜
      inst✝ : OrderTopology 𝕜
      hQ : Ne Q 0
      h : Filter.Tendsto (fun x => HDiv.hDiv (Polynomial.eval x P) (Polynomial.eval  …
      hPQ : Or (Eq P.leadingCoeff 0) (Eq Q.leadingCoeff 0)
      ⊢ LT.lt P.degree Q.degree
    -/
    cases' hPQ with hP0 hQ0
      /-
        case pos.inl
        𝕜 : Type u_1
        inst✝¹ : NormedLinearOrderedField 𝕜
        P Q : Polynomial 𝕜
        inst✝ : OrderTopology 𝕜
        hQ : Ne Q 0
        h : Filter.Tendsto (fun x => HDiv.hDiv (Polynomial.eval x P) (Polynomial.eval  …
        hP0 : Eq P.leadingCoeff 0
        ⊢ LT.lt P.degree Q.degree
      -/
    · rw [leadingCoeff_eq_zero.1 hP0, degree_zero]
      /-
        case pos.inl
        𝕜 : Type u_1
        inst✝¹ : NormedLinearOrderedField 𝕜
        P Q : Polynomial 𝕜
        inst✝ : OrderTopology 𝕜
        hQ : Ne Q 0
        h : Filter.Tendsto (fun x => HDiv.hDiv (Polynomial.eval x P) (Polynomial.eval  …
        hP0 : Eq P.leadingCoeff 0
        ⊢ LT.lt Bot.bot Q.degree
      -/
      exact bot_lt_iff_ne_bot.2 fun hQ' => hQ (degree_eq_bot.1 hQ')
      /-
        🎉 no goals
      -/
      /-
        case pos.inr
        𝕜 : Type u_1
        inst✝¹ : NormedLinearOrderedField 𝕜
        P Q : Polynomial 𝕜
        inst✝ : OrderTopology 𝕜
        hQ : Ne Q 0
        h : Filter.Tendsto (fun x => HDiv.hDiv (Polynomial.eval x P) (Polynomial.eval  …
        hQ0 : Eq Q.leadingCoeff 0
        ⊢ LT.lt P.degree Q.degree
      -/
    · exact absurd (leadingCoeff_eq_zero.1 hQ0) hQ
      /-
        🎉 no goals
      -/
    /-
      case neg
      𝕜 : Type u_1
      inst✝¹ : NormedLinearOrderedField 𝕜
      P Q : Polynomial 𝕜
      inst✝ : OrderTopology 𝕜
      hQ : Ne Q 0
      h : Filter.Tendsto (fun x => HDiv.hDiv (Polynomial.eval x P) (Polynomial.eval  …
      hPQ : Not (Eq (HDiv.hDiv P.leadingCoeff Q.leadingCoeff) 0)
      ⊢ LT.lt P.degree Q.degree
    -/
  · have := (isEquivalent_atTop_div P Q).tendsto_nhds h
    /-
      case neg
      𝕜 : Type u_1
      inst✝¹ : NormedLinearOrderedField 𝕜
      P Q : Polynomial 𝕜
      inst✝ : OrderTopology 𝕜
      hQ : Ne Q 0
      h : Filter.Tendsto (fun x => HDiv.hDiv (Polynomial.eval x P) (Polynomial.eval  …
      hPQ : Not (Eq (HDiv.hDiv P.leadingCoeff Q.leadingCoeff) 0)
      this : Filter.Tendsto (fun x => HMul.hMul (HDiv.hDiv P.leadingCoeff Q.leadingC …
      ⊢ LT.lt P.degree Q.degree
    -/
    rw [tendsto_const_mul_zpow_atTop_nhds_iff hPQ] at this
    /-
      case neg
      𝕜 : Type u_1
      inst✝¹ : NormedLinearOrderedField 𝕜
      P Q : Polynomial 𝕜
      inst✝ : OrderTopology 𝕜
      hQ : Ne Q 0
      h : Filter.Tendsto (fun x => HDiv.hDiv (Polynomial.eval x P) (Polynomial.eval  …
      hPQ : Not (Eq (HDiv.hDiv P.leadingCoeff Q.leadingCoeff) 0)
      this : Or (And (Eq (HSub.hSub ↑P.natDegree ↑Q.natDegree) 0) (Eq (HDiv.hDiv P.l …
      ⊢ LT.lt P.degree Q.degree
    -/
    cases' this with h h
      /-
        case neg.inl
        𝕜 : Type u_1
        inst✝¹ : NormedLinearOrderedField 𝕜
        P Q : Polynomial 𝕜
        inst✝ : OrderTopology 𝕜
        hQ : Ne Q 0
        h✝ : Filter.Tendsto (fun x => HDiv.hDiv (Polynomial.eval x P) (Polynomial.eval …
        hPQ : Not (Eq (HDiv.hDiv P.leadingCoeff Q.leadingCoeff) 0)
        h : And (Eq (HSub.hSub ↑P.natDegree ↑Q.natDegree) 0) (Eq (HDiv.hDiv P.leadingC …
        ⊢ LT.lt P.degree Q.degree
      -/
    · exact absurd h.2 hPQ
      /-
        🎉 no goals
      -/
      /-
        case neg.inr
        𝕜 : Type u_1
        inst✝¹ : NormedLinearOrderedField 𝕜
        P Q : Polynomial 𝕜
        inst✝ : OrderTopology 𝕜
        hQ : Ne Q 0
        h✝ : Filter.Tendsto (fun x => HDiv.hDiv (Polynomial.eval x P) (Polynomial.eval …
        hPQ : Not (Eq (HDiv.hDiv P.leadingCoeff Q.leadingCoeff) 0)
        h : And (LT.lt (HSub.hSub ↑P.natDegree ↑Q.natDegree) 0) (Eq 0 0)
        ⊢ LT.lt P.degree Q.degree
      -/
    · rw [sub_lt_iff_lt_add, zero_add, Int.ofNat_lt] at h
      /-
        case neg.inr
        𝕜 : Type u_1
        inst✝¹ : NormedLinearOrderedField 𝕜
        P Q : Polynomial 𝕜
        inst✝ : OrderTopology 𝕜
        hQ : Ne Q 0
        h✝ : Filter.Tendsto (fun x => HDiv.hDiv (Polynomial.eval x P) (Polynomial.eval …
        hPQ : Not (Eq (HDiv.hDiv P.leadingCoeff Q.leadingCoeff) 0)
        h : And (LT.lt P.natDegree Q.natDegree) (Eq 0 0)
        ⊢ LT.lt P.degree Q.degree
      -/
      exact degree_lt_degree h.1
      /-
        🎉 no goals
      -/


theorem div_tendsto_leadingCoeff_div_of_degree_eq (hdeg : P.degree = Q.degree) :
    Tendsto (fun x => eval x P / eval x Q) atTop (𝓝 <| P.leadingCoeff / Q.leadingCoeff) := by
  /-
    𝕜 : Type u_1
    inst✝¹ : NormedLinearOrderedField 𝕜
    P Q : Polynomial 𝕜
    inst✝ : OrderTopology 𝕜
    hdeg : Eq P.degree Q.degree
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (Polynomial.eval x P) (Polynomial.eval x  …
  -/
  refine (isEquivalent_atTop_div P Q).symm.tendsto_nhds ?_
  /-
    𝕜 : Type u_1
    inst✝¹ : NormedLinearOrderedField 𝕜
    P Q : Polynomial 𝕜
    inst✝ : OrderTopology 𝕜
    hdeg : Eq P.degree Q.degree
    ⊢ Filter.Tendsto (fun x => HMul.hMul (HDiv.hDiv P.leadingCoeff Q.leadingCoeff) …
  -/
  rw [show (P.natDegree : ℤ) = Q.natDegree by simp [hdeg, natDegree]]
  /-
    𝕜 : Type u_1
    inst✝¹ : NormedLinearOrderedField 𝕜
    P Q : Polynomial 𝕜
    inst✝ : OrderTopology 𝕜
    hdeg : Eq P.degree Q.degree
    ⊢ Filter.Tendsto (fun x => HMul.hMul (HDiv.hDiv P.leadingCoeff Q.leadingCoeff) …
  -/
  simp [tendsto_const_nhds]
  /-
    🎉 no goals
  -/


theorem div_tendsto_atTop_of_degree_gt' (hdeg : Q.degree < P.degree)
    (hpos : 0 < P.leadingCoeff / Q.leadingCoeff) :
    Tendsto (fun x => eval x P / eval x Q) atTop atTop := by
  have hQ : Q ≠ 0 := fun h => by
    simp only [h, div_zero, leadingCoeff_zero] at hpos
    exact hpos.false
  /-
    𝕜 : Type u_1
    inst✝¹ : NormedLinearOrderedField 𝕜
    P Q : Polynomial 𝕜
    inst✝ : OrderTopology 𝕜
    hdeg : LT.lt Q.degree P.degree
    hpos : LT.lt 0 (HDiv.hDiv P.leadingCoeff Q.leadingCoeff)
    hQ : Ne Q 0
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (Polynomial.eval x P) (Polynomial.eval x  …
  -/
  rw [← natDegree_lt_natDegree_iff hQ] at hdeg
  /-
    𝕜 : Type u_1
    inst✝¹ : NormedLinearOrderedField 𝕜
    P Q : Polynomial 𝕜
    inst✝ : OrderTopology 𝕜
    hdeg : LT.lt Q.natDegree P.natDegree
    hpos : LT.lt 0 (HDiv.hDiv P.leadingCoeff Q.leadingCoeff)
    hQ : Ne Q 0
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (Polynomial.eval x P) (Polynomial.eval x  …
  -/
  refine (isEquivalent_atTop_div P Q).symm.tendsto_atTop ?_
  /-
    𝕜 : Type u_1
    inst✝¹ : NormedLinearOrderedField 𝕜
    P Q : Polynomial 𝕜
    inst✝ : OrderTopology 𝕜
    hdeg : LT.lt Q.natDegree P.natDegree
    hpos : LT.lt 0 (HDiv.hDiv P.leadingCoeff Q.leadingCoeff)
    hQ : Ne Q 0
    ⊢ Filter.Tendsto (fun x => HMul.hMul (HDiv.hDiv P.leadingCoeff Q.leadingCoeff) …
  -/
  apply Tendsto.const_mul_atTop hpos
  /-
    𝕜 : Type u_1
    inst✝¹ : NormedLinearOrderedField 𝕜
    P Q : Polynomial 𝕜
    inst✝ : OrderTopology 𝕜
    hdeg : LT.lt Q.natDegree P.natDegree
    hpos : LT.lt 0 (HDiv.hDiv P.leadingCoeff Q.leadingCoeff)
    hQ : Ne Q 0
    ⊢ Filter.Tendsto (fun x => HPow.hPow x (HSub.hSub ↑P.natDegree ↑Q.natDegree))  …
  -/
  apply tendsto_zpow_atTop_atTop
  /-
    case hn
    𝕜 : Type u_1
    inst✝¹ : NormedLinearOrderedField 𝕜
    P Q : Polynomial 𝕜
    inst✝ : OrderTopology 𝕜
    hdeg : LT.lt Q.natDegree P.natDegree
    hpos : LT.lt 0 (HDiv.hDiv P.leadingCoeff Q.leadingCoeff)
    hQ : Ne Q 0
    ⊢ LT.lt 0 (HSub.hSub ↑P.natDegree ↑Q.natDegree)
  -/
  omega
  /-
    🎉 no goals
  -/


theorem div_tendsto_atTop_of_degree_gt (hdeg : Q.degree < P.degree) (hQ : Q ≠ 0)
    (hnng : 0 ≤ P.leadingCoeff / Q.leadingCoeff) :
    Tendsto (fun x => eval x P / eval x Q) atTop atTop :=
  have ratio_pos : 0 < P.leadingCoeff / Q.leadingCoeff :=
    lt_of_le_of_ne hnng
      (div_ne_zero (fun h => ne_zero_of_degree_gt hdeg <| leadingCoeff_eq_zero.mp h) fun h =>
          hQ <| leadingCoeff_eq_zero.mp h).symm
  div_tendsto_atTop_of_degree_gt' P Q hdeg ratio_pos


theorem div_tendsto_atBot_of_degree_gt' (hdeg : Q.degree < P.degree)
    (hneg : P.leadingCoeff / Q.leadingCoeff < 0) :
    Tendsto (fun x => eval x P / eval x Q) atTop atBot := by
  have hQ : Q ≠ 0 := fun h => by
    simp only [h, div_zero, leadingCoeff_zero] at hneg
    exact hneg.false
  /-
    𝕜 : Type u_1
    inst✝¹ : NormedLinearOrderedField 𝕜
    P Q : Polynomial 𝕜
    inst✝ : OrderTopology 𝕜
    hdeg : LT.lt Q.degree P.degree
    hneg : LT.lt (HDiv.hDiv P.leadingCoeff Q.leadingCoeff) 0
    hQ : Ne Q 0
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (Polynomial.eval x P) (Polynomial.eval x  …
  -/
  rw [← natDegree_lt_natDegree_iff hQ] at hdeg
  /-
    𝕜 : Type u_1
    inst✝¹ : NormedLinearOrderedField 𝕜
    P Q : Polynomial 𝕜
    inst✝ : OrderTopology 𝕜
    hdeg : LT.lt Q.natDegree P.natDegree
    hneg : LT.lt (HDiv.hDiv P.leadingCoeff Q.leadingCoeff) 0
    hQ : Ne Q 0
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (Polynomial.eval x P) (Polynomial.eval x  …
  -/
  refine (isEquivalent_atTop_div P Q).symm.tendsto_atBot ?_
  /-
    𝕜 : Type u_1
    inst✝¹ : NormedLinearOrderedField 𝕜
    P Q : Polynomial 𝕜
    inst✝ : OrderTopology 𝕜
    hdeg : LT.lt Q.natDegree P.natDegree
    hneg : LT.lt (HDiv.hDiv P.leadingCoeff Q.leadingCoeff) 0
    hQ : Ne Q 0
    ⊢ Filter.Tendsto (fun x => HMul.hMul (HDiv.hDiv P.leadingCoeff Q.leadingCoeff) …
  -/
  apply Tendsto.const_mul_atTop_of_neg hneg
  /-
    𝕜 : Type u_1
    inst✝¹ : NormedLinearOrderedField 𝕜
    P Q : Polynomial 𝕜
    inst✝ : OrderTopology 𝕜
    hdeg : LT.lt Q.natDegree P.natDegree
    hneg : LT.lt (HDiv.hDiv P.leadingCoeff Q.leadingCoeff) 0
    hQ : Ne Q 0
    ⊢ Filter.Tendsto (fun x => HPow.hPow x (HSub.hSub ↑P.natDegree ↑Q.natDegree))  …
  -/
  apply tendsto_zpow_atTop_atTop
  /-
    case hn
    𝕜 : Type u_1
    inst✝¹ : NormedLinearOrderedField 𝕜
    P Q : Polynomial 𝕜
    inst✝ : OrderTopology 𝕜
    hdeg : LT.lt Q.natDegree P.natDegree
    hneg : LT.lt (HDiv.hDiv P.leadingCoeff Q.leadingCoeff) 0
    hQ : Ne Q 0
    ⊢ LT.lt 0 (HSub.hSub ↑P.natDegree ↑Q.natDegree)
  -/
  omega
  /-
    🎉 no goals
  -/


theorem div_tendsto_atBot_of_degree_gt (hdeg : Q.degree < P.degree) (hQ : Q ≠ 0)
    (hnps : P.leadingCoeff / Q.leadingCoeff ≤ 0) :
    Tendsto (fun x => eval x P / eval x Q) atTop atBot :=
  have ratio_neg : P.leadingCoeff / Q.leadingCoeff < 0 :=
    lt_of_le_of_ne hnps
      (div_ne_zero (fun h => ne_zero_of_degree_gt hdeg <| leadingCoeff_eq_zero.mp h) fun h =>
        hQ <| leadingCoeff_eq_zero.mp h)
  div_tendsto_atBot_of_degree_gt' P Q hdeg ratio_neg


theorem abs_div_tendsto_atTop_of_degree_gt (hdeg : Q.degree < P.degree) (hQ : Q ≠ 0) :
    Tendsto (fun x => |eval x P / eval x Q|) atTop atTop := by
  /-
    𝕜 : Type u_1
    inst✝¹ : NormedLinearOrderedField 𝕜
    P Q : Polynomial 𝕜
    inst✝ : OrderTopology 𝕜
    hdeg : LT.lt Q.degree P.degree
    hQ : Ne Q 0
    ⊢ Filter.Tendsto (fun x => abs (HDiv.hDiv (Polynomial.eval x P) (Polynomial.ev …
  -/
  by_cases h : 0 ≤ P.leadingCoeff / Q.leadingCoeff
    /-
      case pos
      𝕜 : Type u_1
      inst✝¹ : NormedLinearOrderedField 𝕜
      P Q : Polynomial 𝕜
      inst✝ : OrderTopology 𝕜
      hdeg : LT.lt Q.degree P.degree
      hQ : Ne Q 0
      h : LE.le 0 (HDiv.hDiv P.leadingCoeff Q.leadingCoeff)
      ⊢ Filter.Tendsto (fun x => abs (HDiv.hDiv (Polynomial.eval x P) (Polynomial.ev …
    -/
  · exact tendsto_abs_atTop_atTop.comp (P.div_tendsto_atTop_of_degree_gt Q hdeg hQ h)
    /-
      🎉 no goals
    -/
    /-
      case neg
      𝕜 : Type u_1
      inst✝¹ : NormedLinearOrderedField 𝕜
      P Q : Polynomial 𝕜
      inst✝ : OrderTopology 𝕜
      hdeg : LT.lt Q.degree P.degree
      hQ : Ne Q 0
      h : Not (LE.le 0 (HDiv.hDiv P.leadingCoeff Q.leadingCoeff))
      ⊢ Filter.Tendsto (fun x => abs (HDiv.hDiv (Polynomial.eval x P) (Polynomial.ev …
    -/
  · push_neg at h
    /-
      case neg
      𝕜 : Type u_1
      inst✝¹ : NormedLinearOrderedField 𝕜
      P Q : Polynomial 𝕜
      inst✝ : OrderTopology 𝕜
      hdeg : LT.lt Q.degree P.degree
      hQ : Ne Q 0
      h : LT.lt (HDiv.hDiv P.leadingCoeff Q.leadingCoeff) 0
      ⊢ Filter.Tendsto (fun x => abs (HDiv.hDiv (Polynomial.eval x P) (Polynomial.ev …
    -/
    exact tendsto_abs_atBot_atTop.comp (P.div_tendsto_atBot_of_degree_gt Q hdeg hQ h.le)
    /-
      🎉 no goals
    -/


theorem isBigO_of_degree_le (h : P.degree ≤ Q.degree) :
    (fun x => eval x P) =O[atTop] fun x => eval x Q := by
  /-
    𝕜 : Type u_1
    inst✝¹ : NormedLinearOrderedField 𝕜
    P Q : Polynomial 𝕜
    inst✝ : OrderTopology 𝕜
    h : LE.le P.degree Q.degree
    ⊢ Asymptotics.IsBigO Filter.atTop (fun x => Polynomial.eval x P) fun x => Poly …
  -/
  by_cases hp : P = 0
    /-
      case pos
      𝕜 : Type u_1
      inst✝¹ : NormedLinearOrderedField 𝕜
      P Q : Polynomial 𝕜
      inst✝ : OrderTopology 𝕜
      h : LE.le P.degree Q.degree
      hp : Eq P 0
      ⊢ Asymptotics.IsBigO Filter.atTop (fun x => Polynomial.eval x P) fun x => Poly …
    -/
  · simpa [hp] using isBigO_zero (fun x => eval x Q) atTop
    /-
      🎉 no goals
    -/
    /-
      case neg
      𝕜 : Type u_1
      inst✝¹ : NormedLinearOrderedField 𝕜
      P Q : Polynomial 𝕜
      inst✝ : OrderTopology 𝕜
      h : LE.le P.degree Q.degree
      hp : Not (Eq P 0)
      ⊢ Asymptotics.IsBigO Filter.atTop (fun x => Polynomial.eval x P) fun x => Poly …
    -/
  · have hq : Q ≠ 0 := ne_zero_of_degree_ge_degree h hp
    have hPQ : ∀ᶠ x : 𝕜 in atTop, eval x Q = 0 → eval x P = 0 :=
      Filter.mem_of_superset (Polynomial.eventually_no_roots Q hq) fun x h h' => absurd h' h
    /-
      case neg
      𝕜 : Type u_1
      inst✝¹ : NormedLinearOrderedField 𝕜
      P Q : Polynomial 𝕜
      inst✝ : OrderTopology 𝕜
      h : LE.le P.degree Q.degree
      hp : Not (Eq P 0)
      hq : Ne Q 0
      hPQ : Filter.Eventually (fun x => Eq (Polynomial.eval x Q) 0 → Eq (Polynomial. …
      ⊢ Asymptotics.IsBigO Filter.atTop (fun x => Polynomial.eval x P) fun x => Poly …
    -/
    cases' le_iff_lt_or_eq.mp h with h h
      /-
        case neg.inl
        𝕜 : Type u_1
        inst✝¹ : NormedLinearOrderedField 𝕜
        P Q : Polynomial 𝕜
        inst✝ : OrderTopology 𝕜
        h✝ : LE.le P.degree Q.degree
        hp : Not (Eq P 0)
        hq : Ne Q 0
        hPQ : Filter.Eventually (fun x => Eq (Polynomial.eval x Q) 0 → Eq (Polynomial. …
        h : LT.lt P.degree Q.degree
        ⊢ Asymptotics.IsBigO Filter.atTop (fun x => Polynomial.eval x P) fun x => Poly …
      -/
    · exact isBigO_of_div_tendsto_nhds hPQ 0 (div_tendsto_zero_of_degree_lt P Q h)
      /-
        🎉 no goals
      -/
      /-
        case neg.inr
        𝕜 : Type u_1
        inst✝¹ : NormedLinearOrderedField 𝕜
        P Q : Polynomial 𝕜
        inst✝ : OrderTopology 𝕜
        h✝ : LE.le P.degree Q.degree
        hp : Not (Eq P 0)
        hq : Ne Q 0
        hPQ : Filter.Eventually (fun x => Eq (Polynomial.eval x Q) 0 → Eq (Polynomial. …
        h : Eq P.degree Q.degree
        ⊢ Asymptotics.IsBigO Filter.atTop (fun x => Polynomial.eval x P) fun x => Poly …
      -/
    · exact isBigO_of_div_tendsto_nhds hPQ _ (div_tendsto_leadingCoeff_div_of_degree_eq P Q h)
      /-
        🎉 no goals
      -/


