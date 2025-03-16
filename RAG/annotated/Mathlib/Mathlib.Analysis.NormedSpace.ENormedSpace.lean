/-- Extended norm on a vector space. As in the case of normed spaces, we require only
`‖c • x‖ ≤ ‖c‖ * ‖x‖` in the definition, then prove an equality in `map_smul`. -/
structure ENormedSpace (𝕜 : Type*) (V : Type*) [NormedField 𝕜] [AddCommGroup V] [Module 𝕜 V] where
  /-- the norm of an ENormedSpace, taking values into `ℝ≥0∞` -/
  toFun : V → ℝ≥0∞
  eq_zero' : ∀ x, toFun x = 0 → x = 0
  map_add_le' : ∀ x y : V, toFun (x + y) ≤ toFun x + toFun y
  map_smul_le' : ∀ (c : 𝕜) (x : V), toFun (c • x) ≤ ‖c‖₊ * toFun x


instance : CoeFun (ENormedSpace 𝕜 V) fun _ => V → ℝ≥0∞ :=
  ⟨ENormedSpace.toFun⟩


theorem coeFn_injective : Function.Injective ((↑) : ENormedSpace 𝕜 V → V → ℝ≥0∞) := by
  /-
    𝕜 : Type u_1
    V : Type u_2
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup V
    inst✝ : Module 𝕜 V
    ⊢ Function.Injective ENormedSpace.toFun
  -/
  intro e₁ e₂ h
  /-
    𝕜 : Type u_1
    V : Type u_2
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup V
    inst✝ : Module 𝕜 V
    e₁ e₂ : ENormedSpace 𝕜 V
    h : Eq ↑e₁ ↑e₂
    ⊢ Eq e₁ e₂
  -/
  cases e₁
  /-
    case mk
    𝕜 : Type u_1
    V : Type u_2
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup V
    inst✝ : Module 𝕜 V
    e₂ : ENormedSpace 𝕜 V
    toFun✝ : V → ENNReal
    eq_zero'✝ : ∀ (x : V), Eq (toFun✝ x) 0 → Eq x 0
    map_add_le'✝ : ∀ (x y : V), LE.le (toFun✝ (HAdd.hAdd x y)) (HAdd.hAdd (toFun✝  …
    map_smul_le'✝ : ∀ (c : 𝕜) (x : V), LE.le (toFun✝ (HSMul.hSMul c x)) (HMul.hMul …
    h : Eq ↑{ toFun := toFun✝, eq_zero' := eq_zero'✝, map_add_le' := map_add_le'✝, …
    ⊢ Eq { toFun := toFun✝, eq_zero' := eq_zero'✝, map_add_le' := map_add_le'✝, ma …
  -/
  cases e₂
  /-
    case mk.mk
    𝕜 : Type u_1
    V : Type u_2
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup V
    inst✝ : Module 𝕜 V
    toFun✝¹ : V → ENNReal
    eq_zero'✝¹ : ∀ (x : V), Eq (toFun✝¹ x) 0 → Eq x 0
    map_add_le'✝¹ : ∀ (x y : V), LE.le (toFun✝¹ (HAdd.hAdd x y)) (HAdd.hAdd (toFun …
    map_smul_le'✝¹ : ∀ (c : 𝕜) (x : V), LE.le (toFun✝¹ (HSMul.hSMul c x)) (HMul.hM …
    toFun✝ : V → ENNReal
    eq_zero'✝ : ∀ (x : V), Eq (toFun✝ x) 0 → Eq x 0
    map_add_le'✝ : ∀ (x y : V), LE.le (toFun✝ (HAdd.hAdd x y)) (HAdd.hAdd (toFun✝  …
    map_smul_le'✝ : ∀ (c : 𝕜) (x : V), LE.le (toFun✝ (HSMul.hSMul c x)) (HMul.hMul …
    h : Eq ↑{ toFun := toFun✝¹, eq_zero' := eq_zero'✝¹, map_add_le' := map_add_le' …
    ⊢ Eq { toFun := toFun✝¹, eq_zero' := eq_zero'✝¹, map_add_le' := map_add_le'✝¹, …
  -/
  congr
  /-
    🎉 no goals
  -/


@[ext]
theorem ext {e₁ e₂ : ENormedSpace 𝕜 V} (h : ∀ x, e₁ x = e₂ x) : e₁ = e₂ :=
  coeFn_injective <| funext h


@[simp, norm_cast]
theorem coe_inj {e₁ e₂ : ENormedSpace 𝕜 V} : (e₁ : V → ℝ≥0∞) = e₂ ↔ e₁ = e₂ :=
  coeFn_injective.eq_iff


@[simp]
theorem map_smul (c : 𝕜) (x : V) : e (c • x) = ‖c‖₊ * e x := by
  /-
    𝕜 : Type u_1
    V : Type u_2
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup V
    inst✝ : Module 𝕜 V
    e : ENormedSpace 𝕜 V
    c : 𝕜
    x : V
    ⊢ Eq (↑e (HSMul.hSMul c x)) (HMul.hMul (↑(NNNorm.nnnorm c)) (↑e x))
  -/
  apply le_antisymm (e.map_smul_le' c x)
  /-
    𝕜 : Type u_1
    V : Type u_2
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup V
    inst✝ : Module 𝕜 V
    e : ENormedSpace 𝕜 V
    c : 𝕜
    x : V
    ⊢ LE.le (HMul.hMul (↑(NNNorm.nnnorm c)) (↑e x)) (↑e (HSMul.hSMul c x))
  -/
  by_cases hc : c = 0
    /-
      case pos
      𝕜 : Type u_1
      V : Type u_2
      inst✝² : NormedField 𝕜
      inst✝¹ : AddCommGroup V
      inst✝ : Module 𝕜 V
      e : ENormedSpace 𝕜 V
      c : 𝕜
      x : V
      hc : Eq c 0
      ⊢ LE.le (HMul.hMul (↑(NNNorm.nnnorm c)) (↑e x)) (↑e (HSMul.hSMul c x))
    -/
  · simp [hc]
    /-
      🎉 no goals
    -/
  calc
    (‖c‖₊ : ℝ≥0∞) * e x = ‖c‖₊ * e (c⁻¹ • c • x) := by rw [inv_smul_smul₀ hc]
    _ ≤ ‖c‖₊ * (‖c⁻¹‖₊ * e (c • x)) := mul_le_mul_left' (e.map_smul_le' _ _) _
    _ = e (c • x) := by
      rw [← mul_assoc, nnnorm_inv, ENNReal.coe_inv, ENNReal.mul_inv_cancel _ ENNReal.coe_ne_top,
        one_mul]
        <;> simp [hc]


@[simp]
theorem map_zero : e 0 = 0 := by
  /-
    𝕜 : Type u_1
    V : Type u_2
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup V
    inst✝ : Module 𝕜 V
    e : ENormedSpace 𝕜 V
    ⊢ Eq (↑e 0) 0
  -/
  rw [← zero_smul 𝕜 (0 : V), e.map_smul]
  /-
    𝕜 : Type u_1
    V : Type u_2
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup V
    inst✝ : Module 𝕜 V
    e : ENormedSpace 𝕜 V
    ⊢ Eq (HMul.hMul (↑(NNNorm.nnnorm 0)) (↑e 0)) 0
  -/
  norm_num
  /-
    🎉 no goals
  -/


@[simp]
theorem eq_zero_iff {x : V} : e x = 0 ↔ x = 0 :=
  ⟨e.eq_zero' x, fun h => h.symm ▸ e.map_zero⟩


@[simp]
theorem map_neg (x : V) : e (-x) = e x :=
  calc
                                     /-
                                       𝕜 : Type u_1
                                       V : Type u_2
                                       inst✝² : NormedField 𝕜
                                       inst✝¹ : AddCommGroup V
                                       inst✝ : Module 𝕜 V
                                       e : ENormedSpace 𝕜 V
                                       x : V
                                       ⊢ Eq (↑e (Neg.neg x)) (HMul.hMul (↑(NNNorm.nnnorm (-1))) (↑e x))
                                     -/
    e (-x) = ‖(-1 : 𝕜)‖₊ * e x := by rw [← map_smul, neg_one_smul]
                                     /-
                                       🎉 no goals
                                     -/
                  /-
                    𝕜 : Type u_1
                    V : Type u_2
                    inst✝² : NormedField 𝕜
                    inst✝¹ : AddCommGroup V
                    inst✝ : Module 𝕜 V
                    e : ENormedSpace 𝕜 V
                    x : V
                    ⊢ Eq (HMul.hMul (↑(NNNorm.nnnorm (-1))) (↑e x)) (↑e x)
                  -/
    _ = e x := by simp
                  /-
                    🎉 no goals
                  -/


                                                            /-
                                                              𝕜 : Type u_1
                                                              V : Type u_2
                                                              inst✝² : NormedField 𝕜
                                                              inst✝¹ : AddCommGroup V
                                                              inst✝ : Module 𝕜 V
                                                              e : ENormedSpace 𝕜 V
                                                              x y : V
                                                              ⊢ Eq (↑e (HSub.hSub x y)) (↑e (HSub.hSub y x))
                                                            -/
theorem map_sub_rev (x y : V) : e (x - y) = e (y - x) := by rw [← neg_sub, e.map_neg]
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem map_add_le (x y : V) : e (x + y) ≤ e x + e y :=
  e.map_add_le' x y


theorem map_sub_le (x y : V) : e (x - y) ≤ e x + e y :=
  calc
                                 /-
                                   𝕜 : Type u_1
                                   V : Type u_2
                                   inst✝² : NormedField 𝕜
                                   inst✝¹ : AddCommGroup V
                                   inst✝ : Module 𝕜 V
                                   e : ENormedSpace 𝕜 V
                                   x y : V
                                   ⊢ Eq (↑e (HSub.hSub x y)) (↑e (HAdd.hAdd x (Neg.neg y)))
                                 -/
    e (x - y) = e (x + -y) := by rw [sub_eq_add_neg]
                                 /-
                                   🎉 no goals
                                 -/
    _ ≤ e x + e (-y) := e.map_add_le x (-y)
                        /-
                          𝕜 : Type u_1
                          V : Type u_2
                          inst✝² : NormedField 𝕜
                          inst✝¹ : AddCommGroup V
                          inst✝ : Module 𝕜 V
                          e : ENormedSpace 𝕜 V
                          x y : V
                          ⊢ Eq (HAdd.hAdd (↑e x) (↑e (Neg.neg y))) (HAdd.hAdd (↑e x) (↑e y))
                        -/
    _ = e x + e y := by rw [e.map_neg]
                        /-
                          🎉 no goals
                        -/


instance partialOrder : PartialOrder (ENormedSpace 𝕜 V) where
  le e₁ e₂ := ∀ x, e₁ x ≤ e₂ x
  le_refl _ _ := le_rfl
  le_trans _ _ _ h₁₂ h₂₃ x := le_trans (h₁₂ x) (h₂₃ x)
  le_antisymm _ _ h₁₂ h₂₁ := ext fun x => le_antisymm (h₁₂ x) (h₂₁ x)


/-- The `ENormedSpace` sending each non-zero vector to infinity. -/
noncomputable instance : Top (ENormedSpace 𝕜 V) :=
  ⟨{  toFun := fun x => if x = 0 then 0 else ⊤
                              /-
                                𝕜 : Type u_1
                                V : Type u_2
                                inst✝² : NormedField 𝕜
                                inst✝¹ : AddCommGroup V
                                inst✝ : Module 𝕜 V
                                e : ENormedSpace 𝕜 V
                                x : V
                                ⊢ Eq ((fun x => ite (Eq x 0) 0 Top.top) x) 0 → Eq x 0
                              -/
                                                       /-
                                                         🎉 no goals
                                                       -/
      eq_zero' := fun x => by simp only; split_ifs <;> simp [*]
                                                       /-
                                                         🎉 no goals
                                                       -/
      map_add_le' := fun x y => by
        /-
          𝕜 : Type u_1
          V : Type u_2
          inst✝² : NormedField 𝕜
          inst✝¹ : AddCommGroup V
          inst✝ : Module 𝕜 V
          e : ENormedSpace 𝕜 V
          x y : V
          ⊢ LE.le ((fun x => ite (Eq x 0) 0 Top.top) (HAdd.hAdd x y)) (HAdd.hAdd ((fun x …
        -/
        simp only
        /-
          𝕜 : Type u_1
          V : Type u_2
          inst✝² : NormedField 𝕜
          inst✝¹ : AddCommGroup V
          inst✝ : Module 𝕜 V
          e : ENormedSpace 𝕜 V
          x y : V
          ⊢ LE.le (ite (Eq (HAdd.hAdd x y) 0) 0 Top.top) (HAdd.hAdd (ite (Eq x 0) 0 Top. …
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
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
        split_ifs with hxy hx hy hy hx hy hy <;> try simp [*]
                                                 /-
                                                   🎉 no goals
                                                 -/
        /-
          case pos
          𝕜 : Type u_1
          V : Type u_2
          inst✝² : NormedField 𝕜
          inst✝¹ : AddCommGroup V
          inst✝ : Module 𝕜 V
          e : ENormedSpace 𝕜 V
          x y : V
          hxy : Not (Eq (HAdd.hAdd x y) 0)
          hx : Eq x 0
          hy : Eq y 0
          ⊢ False
        -/
        simp [hx, hy] at hxy
        /-
          🎉 no goals
        -/
      map_smul_le' := fun c x => by
        /-
          𝕜 : Type u_1
          V : Type u_2
          inst✝² : NormedField 𝕜
          inst✝¹ : AddCommGroup V
          inst✝ : Module 𝕜 V
          e : ENormedSpace 𝕜 V
          c : 𝕜
          x : V
          ⊢ LE.le ((fun x => ite (Eq x 0) 0 Top.top) (HSMul.hSMul c x)) (HMul.hMul (↑(NN …
        -/
        simp only
        /-
          𝕜 : Type u_1
          V : Type u_2
          inst✝² : NormedField 𝕜
          inst✝¹ : AddCommGroup V
          inst✝ : Module 𝕜 V
          e : ENormedSpace 𝕜 V
          c : 𝕜
          x : V
          ⊢ LE.le (ite (Eq (HSMul.hSMul c x) 0) 0 Top.top) (HMul.hMul (↑(NNNorm.nnnorm c …
        -/
        split_ifs with hcx hx hx <;> simp only [smul_eq_zero, not_or] at hcx
          /-
            case pos
            𝕜 : Type u_1
            V : Type u_2
            inst✝² : NormedField 𝕜
            inst✝¹ : AddCommGroup V
            inst✝ : Module 𝕜 V
            e : ENormedSpace 𝕜 V
            c : 𝕜
            x : V
            hx : Eq x 0
            hcx : Or (Eq c 0) (Eq x 0)
            ⊢ LE.le 0 (HMul.hMul (↑(NNNorm.nnnorm c)) 0)
          -/
        · simp only [mul_zero, le_refl]
          /-
            🎉 no goals
          -/
          /-
            case neg
            𝕜 : Type u_1
            V : Type u_2
            inst✝² : NormedField 𝕜
            inst✝¹ : AddCommGroup V
            inst✝ : Module 𝕜 V
            e : ENormedSpace 𝕜 V
            c : 𝕜
            x : V
            hx : Not (Eq x 0)
            hcx : Or (Eq c 0) (Eq x 0)
            ⊢ LE.le 0 (HMul.hMul (↑(NNNorm.nnnorm c)) Top.top)
          -/
        · have : c = 0 := by tauto
          /-
            case neg
            𝕜 : Type u_1
            V : Type u_2
            inst✝² : NormedField 𝕜
            inst✝¹ : AddCommGroup V
            inst✝ : Module 𝕜 V
            e : ENormedSpace 𝕜 V
            c : 𝕜
            x : V
            hx : Not (Eq x 0)
            hcx : Or (Eq c 0) (Eq x 0)
            this : Eq c 0
            ⊢ LE.le 0 (HMul.hMul (↑(NNNorm.nnnorm c)) Top.top)
          -/
          simp [this]
          /-
            🎉 no goals
          -/
          /-
            case pos
            𝕜 : Type u_1
            V : Type u_2
            inst✝² : NormedField 𝕜
            inst✝¹ : AddCommGroup V
            inst✝ : Module 𝕜 V
            e : ENormedSpace 𝕜 V
            c : 𝕜
            x : V
            hx : Eq x 0
            hcx : And (Not (Eq c 0)) (Not (Eq x 0))
            ⊢ LE.le Top.top (HMul.hMul (↑(NNNorm.nnnorm c)) 0)
          -/
        · tauto
          /-
            🎉 no goals
          -/
          /-
            case neg
            𝕜 : Type u_1
            V : Type u_2
            inst✝² : NormedField 𝕜
            inst✝¹ : AddCommGroup V
            inst✝ : Module 𝕜 V
            e : ENormedSpace 𝕜 V
            c : 𝕜
            x : V
            hx : Not (Eq x 0)
            hcx : And (Not (Eq c 0)) (Not (Eq x 0))
            ⊢ LE.le Top.top (HMul.hMul (↑(NNNorm.nnnorm c)) Top.top)
          -/
        · simpa [mul_top'] using hcx.1 }⟩
          /-
            🎉 no goals
          -/


noncomputable instance : Inhabited (ENormedSpace 𝕜 V) :=
  ⟨⊤⟩


theorem top_map {x : V} (hx : x ≠ 0) : (⊤ : ENormedSpace 𝕜 V) x = ⊤ :=
  if_neg hx


noncomputable instance : OrderTop (ENormedSpace 𝕜 V) where
  top := ⊤
                                     /-
                                       𝕜 : Type u_1
                                       V : Type u_2
                                       inst✝² : NormedField 𝕜
                                       inst✝¹ : AddCommGroup V
                                       inst✝ : Module 𝕜 V
                                       e✝ e : ENormedSpace 𝕜 V
                                       x : V
                                       h : Eq x 0
                                       ⊢ LE.le (↑e x) (↑Top.top x)
                                     -/
                                     /-
                                       🎉 no goals
                                     -/
  le_top e x := if h : x = 0 then by simp [h] else by simp [top_map h]
                                                      /-
                                                        🎉 no goals
                                                      -/


noncomputable instance : SemilatticeSup (ENormedSpace 𝕜 V) :=
  { ENormedSpace.partialOrder with
    le := (· ≤ ·)
    lt := (· < ·)
    sup := fun e₁ e₂ =>
      { toFun := fun x => max (e₁ x) (e₂ x)
        eq_zero' := fun _ h => e₁.eq_zero_iff.1 (ENNReal.max_eq_zero_iff.1 h).1
        map_add_le' := fun _ _ =>
          max_le (le_trans (e₁.map_add_le _ _) <| add_le_add (le_max_left _ _) (le_max_left _ _))
            (le_trans (e₂.map_add_le _ _) <| add_le_add (le_max_right _ _) (le_max_right _ _))
                                                  /-
                                                    𝕜 : Type u_1
                                                    V : Type u_2
                                                    inst✝² : NormedField 𝕜
                                                    inst✝¹ : AddCommGroup V
                                                    inst✝ : Module 𝕜 V
                                                    e e₁ e₂ : ENormedSpace 𝕜 V
                                                    c : 𝕜
                                                    x : V
                                                    ⊢ Eq ((fun x => Max.max (↑e₁ x) (↑e₂ x)) (HSMul.hSMul c x)) (HMul.hMul (↑(NNNo …
                                                  -/
        map_smul_le' := fun c x => le_of_eq <| by simp only [map_smul, mul_max] }
                                                  /-
                                                    🎉 no goals
                                                  -/
    le_sup_left := fun _ _ _ => le_max_left _ _
    le_sup_right := fun _ _ _ => le_max_right _ _
    sup_le := fun _ _ _ h₁ h₂ x => max_le (h₁ x) (h₂ x) }


@[simp, norm_cast]
theorem coe_max (e₁ e₂ : ENormedSpace 𝕜 V) : ⇑(e₁ ⊔ e₂) = fun x => max (e₁ x) (e₂ x) :=
  rfl


@[norm_cast]
theorem max_map (e₁ e₂ : ENormedSpace 𝕜 V) (x : V) : (e₁ ⊔ e₂) x = max (e₁ x) (e₂ x) :=
  rfl


/-- Structure of an `EMetricSpace` defined by an extended norm. -/
abbrev emetricSpace : EMetricSpace V where
  edist x y := e (x - y)
                     /-
                       𝕜 : Type u_1
                       V : Type u_2
                       inst✝² : NormedField 𝕜
                       inst✝¹ : AddCommGroup V
                       inst✝ : Module 𝕜 V
                       e : ENormedSpace 𝕜 V
                       x : V
                       ⊢ Eq (EDist.edist x x) 0
                     -/
  edist_self x := by simp
                     /-
                       🎉 no goals
                     -/
                                  /-
                                    𝕜 : Type u_1
                                    V : Type u_2
                                    inst✝² : NormedField 𝕜
                                    inst✝¹ : AddCommGroup V
                                    inst✝ : Module 𝕜 V
                                    e : ENormedSpace 𝕜 V
                                    x y : V
                                    ⊢ Eq (EDist.edist x y) 0 → Eq x y
                                  -/
  eq_of_edist_eq_zero {x y} := by simp [sub_eq_zero]
                                  /-
                                    🎉 no goals
                                  -/
  edist_comm := e.map_sub_rev
                                            /-
                                              𝕜 : Type u_1
                                              V : Type u_2
                                              inst✝² : NormedField 𝕜
                                              inst✝¹ : AddCommGroup V
                                              inst✝ : Module 𝕜 V
                                              e : ENormedSpace 𝕜 V
                                              x y z : V
                                              ⊢ Eq (↑e (HSub.hSub x z)) (↑e (HAdd.hAdd (HSub.hSub x y) (HSub.hSub y z)))
                                            -/
  edist_triangle x y z :=
                                            /-
                                              🎉 no goals
                                            -/
    calc
      e (x - z) = e (x - y + (y - z)) := by rw [sub_add_sub_cancel]
      _ ≤ e (x - y) + e (y - z) := e.map_add_le (x - y) (y - z)


/-- The subspace of vectors with finite ENormedSpace. -/
def finiteSubspace : Subspace 𝕜 V where
  carrier := { x | e x < ⊤ }
                  /-
                    𝕜 : Type u_1
                    V : Type u_2
                    inst✝² : NormedField 𝕜
                    inst✝¹ : AddCommGroup V
                    inst✝ : Module 𝕜 V
                    e : ENormedSpace 𝕜 V
                    ⊢ Membership.mem { carrier := setOf fun x => LT.lt (↑e x) Top.top, add_mem' := …
                  -/
  zero_mem' := by simp
                  /-
                    🎉 no goals
                  -/
  add_mem' {x y} hx hy := lt_of_le_of_lt (e.map_add_le x y) (ENNReal.add_lt_top.2 ⟨hx, hy⟩)
  smul_mem' c x (hx : _ < _) :=
    calc
      e (c • x) = ‖c‖₊ * e x := e.map_smul c x
      _ < ⊤ := ENNReal.mul_lt_top ENNReal.coe_lt_top hx


/-- Metric space structure on `e.finiteSubspace`. We use `EMetricSpace.toMetricSpace`
to ensure that this definition agrees with `e.emetricSpace`. -/
instance metricSpace : MetricSpace e.finiteSubspace := by
  /-
    𝕜 : Type u_1
    V : Type u_2
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup V
    inst✝ : Module 𝕜 V
    e : ENormedSpace 𝕜 V
    ⊢ MetricSpace (Subtype fun x => Membership.mem e.finiteSubspace x)
  -/
  letI := e.emetricSpace
  /-
    𝕜 : Type u_1
    V : Type u_2
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup V
    inst✝ : Module 𝕜 V
    e : ENormedSpace 𝕜 V
    this : EMetricSpace V := e.emetricSpace
    ⊢ MetricSpace (Subtype fun x => Membership.mem e.finiteSubspace x)
  -/
  refine EMetricSpace.toMetricSpace fun x y => ?_
  /-
    𝕜 : Type u_1
    V : Type u_2
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup V
    inst✝ : Module 𝕜 V
    e : ENormedSpace 𝕜 V
    this : EMetricSpace V := e.emetricSpace
    x y : Subtype fun x => Membership.mem e.finiteSubspace x
    ⊢ Ne (EDist.edist x y) Top.top
  -/
  change e (x - y) ≠ ⊤
  /-
    𝕜 : Type u_1
    V : Type u_2
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup V
    inst✝ : Module 𝕜 V
    e : ENormedSpace 𝕜 V
    this : EMetricSpace V := e.emetricSpace
    x y : Subtype fun x => Membership.mem e.finiteSubspace x
    ⊢ Ne (↑e (HSub.hSub ↑x ↑y)) Top.top
  -/
  exact ne_top_of_le_ne_top (ENNReal.add_lt_top.2 ⟨x.2, y.2⟩).ne (e.map_sub_le x y)
  /-
    🎉 no goals
  -/


theorem finite_dist_eq (x y : e.finiteSubspace) : dist x y = (e (x - y)).toReal :=
  rfl


theorem finite_edist_eq (x y : e.finiteSubspace) : edist x y = e (x - y) :=
  rfl


/-- Normed group instance on `e.finiteSubspace`. -/
instance normedAddCommGroup : NormedAddCommGroup e.finiteSubspace :=
  { e.metricSpace with
    norm := fun x => (e x).toReal
    dist_eq := fun _ _ => rfl }


theorem finite_norm_eq (x : e.finiteSubspace) : ‖x‖ = (e x).toReal :=
  rfl


/-- Normed space instance on `e.finiteSubspace`. -/
instance normedSpace : NormedSpace 𝕜 e.finiteSubspace where
                                     /-
                                       𝕜 : Type u_1
                                       V : Type u_2
                                       inst✝² : NormedField 𝕜
                                       inst✝¹ : AddCommGroup V
                                       inst✝ : Module 𝕜 V
                                       e : ENormedSpace 𝕜 V
                                       c : 𝕜
                                       x : Subtype fun x => Membership.mem e.finiteSubspace x
                                       ⊢ Eq (Norm.norm (HSMul.hSMul c x)) (HMul.hMul (Norm.norm c) (Norm.norm x))
                                     -/
  norm_smul_le c x := le_of_eq <| by simp [finite_norm_eq, ENNReal.toReal_mul]
                                     /-
                                       🎉 no goals
                                     -/


