open Classical in
/-- Auxiliary function used to define `ProbabilityTheory.Kernel.rnDeriv` and
`ProbabilityTheory.Kernel.singularPart`.

This has the properties we want for a Radon-Nikodym derivative only if `κ ≪ ν`. The definition of
`rnDeriv κ η` will be built from `rnDerivAux κ (κ + η)`. -/
noncomputable
def rnDerivAux (κ η : Kernel α γ) (a : α) (x : γ) : ℝ :=
  if hα : Countable α then ((κ a).rnDeriv (η a) x).toReal
  else haveI := hαγ.countableOrCountablyGenerated.resolve_left hα
    density (map κ (fun a ↦ (a, ()))) η a x univ


lemma rnDerivAux_nonneg (hκη : κ ≤ η) {a : α} {x : γ} : 0 ≤ rnDerivAux κ η a x := by
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    κ η : ProbabilityTheory.Kernel α γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    hκη : LE.le κ η
    a : α
    x : γ
    ⊢ LE.le 0 (κ.rnDerivAux η a x)
  -/
  rw [rnDerivAux]
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    κ η : ProbabilityTheory.Kernel α γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    hκη : LE.le κ η
    a : α
    x : γ
    ⊢ LE.le 0 (dite (Countable α) (fun hα => ((κ a).rnDeriv (η a) x).toReal) fun h …
  -/
  split_ifs with hα
    /-
      case pos
      α : Type u_1
      γ : Type u_2
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      κ η : ProbabilityTheory.Kernel α γ
      hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
      hκη : LE.le κ η
      a : α
      x : γ
      hα : Countable α
      ⊢ LE.le 0 ((κ a).rnDeriv (η a) x).toReal
    -/
  · exact ENNReal.toReal_nonneg
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      γ : Type u_2
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      κ η : ProbabilityTheory.Kernel α γ
      hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
      hκη : LE.le κ η
      a : α
      x : γ
      hα : Not (Countable α)
      ⊢ LE.le 0 ((κ.map fun a => { fst := a, snd := Unit.unit }).density η a x Set.u …
    -/
  · have := hαγ.countableOrCountablyGenerated.resolve_left hα
    /-
      case neg
      α : Type u_1
      γ : Type u_2
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      κ η : ProbabilityTheory.Kernel α γ
      hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
      hκη : LE.le κ η
      a : α
      x : γ
      hα : Not (Countable α)
      this : MeasurableSpace.CountablyGenerated γ
      ⊢ LE.le 0 ((κ.map fun a => { fst := a, snd := Unit.unit }).density η a x Set.u …
    -/
    exact density_nonneg ((fst_map_id_prod _ measurable_const).trans_le hκη) _ _ _
    /-
      🎉 no goals
    -/


lemma rnDerivAux_le_one [IsFiniteKernel η] (hκη : κ ≤ η) {a : α} :
    rnDerivAux κ η a ≤ᵐ[η a] 1 := by
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    κ η : ProbabilityTheory.Kernel α γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    hκη : LE.le κ η
    a : α
    ⊢ (MeasureTheory.ae (η a)).EventuallyLE (κ.rnDerivAux η a) 1
  -/
  filter_upwards [Measure.rnDeriv_le_one_of_le (hκη a)] with x hx_le_one
  /-
    case h
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    κ η : ProbabilityTheory.Kernel α γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    hκη : LE.le κ η
    a : α
    x : γ
    hx_le_one : LE.le ((κ a).rnDeriv (η a) x) (1 x)
    ⊢ LE.le (κ.rnDerivAux η a x) (1 x)
  -/
  simp_rw [rnDerivAux]
  /-
    case h
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    κ η : ProbabilityTheory.Kernel α γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    hκη : LE.le κ η
    a : α
    x : γ
    hx_le_one : LE.le ((κ a).rnDeriv (η a) x) (1 x)
    ⊢ LE.le (dite (Countable α) (fun hα => ((κ a).rnDeriv (η a) x).toReal) fun hα  …
  -/
  split_ifs with hα
    /-
      case pos
      α : Type u_1
      γ : Type u_2
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      κ η : ProbabilityTheory.Kernel α γ
      hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
      inst✝ : ProbabilityTheory.IsFiniteKernel η
      hκη : LE.le κ η
      a : α
      x : γ
      hx_le_one : LE.le ((κ a).rnDeriv (η a) x) (1 x)
      hα : Countable α
      ⊢ LE.le ((κ a).rnDeriv (η a) x).toReal (1 x)
    -/
  · refine ENNReal.toReal_le_of_le_ofReal zero_le_one ?_
    /-
      case pos
      α : Type u_1
      γ : Type u_2
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      κ η : ProbabilityTheory.Kernel α γ
      hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
      inst✝ : ProbabilityTheory.IsFiniteKernel η
      hκη : LE.le κ η
      a : α
      x : γ
      hx_le_one : LE.le ((κ a).rnDeriv (η a) x) (1 x)
      hα : Countable α
      ⊢ LE.le ((κ a).rnDeriv (η a) x) (ENNReal.ofReal (1 x))
    -/
    simp only [Pi.one_apply, ENNReal.ofReal_one]
    /-
      case pos
      α : Type u_1
      γ : Type u_2
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      κ η : ProbabilityTheory.Kernel α γ
      hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
      inst✝ : ProbabilityTheory.IsFiniteKernel η
      hκη : LE.le κ η
      a : α
      x : γ
      hx_le_one : LE.le ((κ a).rnDeriv (η a) x) (1 x)
      hα : Countable α
      ⊢ LE.le ((κ a).rnDeriv (η a) x) 1
    -/
    exact hx_le_one
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      γ : Type u_2
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      κ η : ProbabilityTheory.Kernel α γ
      hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
      inst✝ : ProbabilityTheory.IsFiniteKernel η
      hκη : LE.le κ η
      a : α
      x : γ
      hx_le_one : LE.le ((κ a).rnDeriv (η a) x) (1 x)
      hα : Not (Countable α)
      ⊢ LE.le ((κ.map fun a => { fst := a, snd := Unit.unit }).density η a x Set.uni …
    -/
  · have := hαγ.countableOrCountablyGenerated.resolve_left hα
    /-
      case neg
      α : Type u_1
      γ : Type u_2
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      κ η : ProbabilityTheory.Kernel α γ
      hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
      inst✝ : ProbabilityTheory.IsFiniteKernel η
      hκη : LE.le κ η
      a : α
      x : γ
      hx_le_one : LE.le ((κ a).rnDeriv (η a) x) (1 x)
      hα : Not (Countable α)
      this : MeasurableSpace.CountablyGenerated γ
      ⊢ LE.le ((κ.map fun a => { fst := a, snd := Unit.unit }).density η a x Set.uni …
    -/
    exact density_le_one ((fst_map_id_prod _ measurable_const).trans_le hκη) _ _ _
    /-
      🎉 no goals
    -/


lemma measurable_rnDerivAux (κ η : Kernel α γ) :
    Measurable (fun p : α × γ ↦ Kernel.rnDerivAux κ η p.1 p.2) := by
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    ⊢ Measurable fun p => κ.rnDerivAux η p.1 p.2
  -/
  simp_rw [rnDerivAux]
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    ⊢ Measurable fun p => dite (Countable α) (fun hα => ((κ p.1).rnDeriv (η p.1) p …
  -/
  split_ifs with hα
    /-
      case pos
      α : Type u_1
      γ : Type u_2
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
      κ η : ProbabilityTheory.Kernel α γ
      hα : Countable α
      ⊢ Measurable fun p => ((κ p.1).rnDeriv (η p.1) p.2).toReal
    -/
  · refine Measurable.ennreal_toReal ?_
    /-
      case pos
      α : Type u_1
      γ : Type u_2
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
      κ η : ProbabilityTheory.Kernel α γ
      hα : Countable α
      ⊢ Measurable fun p => (κ p.1).rnDeriv (η p.1) p.2
    -/
    change Measurable ((fun q : γ × α ↦ (κ q.2).rnDeriv (η q.2) q.1) ∘ Prod.swap)
    /-
      case pos
      α : Type u_1
      γ : Type u_2
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
      κ η : ProbabilityTheory.Kernel α γ
      hα : Countable α
      ⊢ Measurable (Function.comp (fun q => (κ q.2).rnDeriv (η q.2) q.1) Prod.swap)
    -/
    refine (measurable_from_prod_countable' (fun a ↦ ?_) ?_).comp measurable_swap
      /-
        case pos.refine_1
        α : Type u_1
        γ : Type u_2
        mα : MeasurableSpace α
        mγ : MeasurableSpace γ
        hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
        κ η : ProbabilityTheory.Kernel α γ
        hα : Countable α
        a : α
        ⊢ Measurable fun x => (κ { fst := x, snd := a }.2).rnDeriv (η { fst := x, snd  …
      -/
    · exact Measure.measurable_rnDeriv (κ a) (η a)
      /-
        🎉 no goals
      -/
      /-
        case pos.refine_2
        α : Type u_1
        γ : Type u_2
        mα : MeasurableSpace α
        mγ : MeasurableSpace γ
        hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
        κ η : ProbabilityTheory.Kernel α γ
        hα : Countable α
        ⊢ ∀ (y y' : α) (x : γ), Membership.mem (measurableAtom y) y' → Eq ((κ { fst := …
      -/
    · intro a a' c ha'_mem_a
      have h_eq : ∀ κ : Kernel α γ, κ a' = κ a := fun κ ↦ by
        ext s hs
        exact mem_of_mem_measurableAtom ha'_mem_a
          (Kernel.measurable_coe κ hs (measurableSet_singleton (κ a s))) rfl
      /-
        case pos.refine_2
        α : Type u_1
        γ : Type u_2
        mα : MeasurableSpace α
        mγ : MeasurableSpace γ
        hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
        κ η : ProbabilityTheory.Kernel α γ
        hα : Countable α
        a a' : α
        c : γ
        ha'_mem_a : Membership.mem (measurableAtom a) a'
        h_eq : ∀ (κ : ProbabilityTheory.Kernel α γ), Eq (κ a') (κ a)
        ⊢ Eq ((κ { fst := c, snd := a' }.2).rnDeriv (η { fst := c, snd := a' }.2) { fs …
      -/
      rw [h_eq κ, h_eq η]
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u_1
      γ : Type u_2
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
      κ η : ProbabilityTheory.Kernel α γ
      hα : Not (Countable α)
      ⊢ Measurable fun p => (κ.map fun a => { fst := a, snd := Unit.unit }).density  …
    -/
  · have := hαγ.countableOrCountablyGenerated.resolve_left hα
    /-
      case neg
      α : Type u_1
      γ : Type u_2
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
      κ η : ProbabilityTheory.Kernel α γ
      hα : Not (Countable α)
      this : MeasurableSpace.CountablyGenerated γ
      ⊢ Measurable fun p => (κ.map fun a => { fst := a, snd := Unit.unit }).density  …
    -/
    exact measurable_density _ η MeasurableSet.univ
    /-
      🎉 no goals
    -/


lemma measurable_rnDerivAux_right (κ η : Kernel α γ) (a : α) :
    Measurable (fun x : γ ↦ rnDerivAux κ η a x) := by
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    a : α
    ⊢ Measurable fun x => κ.rnDerivAux η a x
  -/
  change Measurable ((fun p : α × γ ↦ rnDerivAux κ η p.1 p.2) ∘ (fun x ↦ (a, x)))
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    a : α
    ⊢ Measurable (Function.comp (fun p => κ.rnDerivAux η p.1 p.2) fun x => { fst : …
  -/
  exact (measurable_rnDerivAux _ _).comp measurable_prod_mk_left
  /-
    🎉 no goals
  -/


lemma setLIntegral_rnDerivAux (κ η : Kernel α γ) [IsFiniteKernel κ] [IsFiniteKernel η]
    (a : α) {s : Set γ} (hs : MeasurableSet s) :
    ∫⁻ x in s, ENNReal.ofReal (rnDerivAux κ (κ + η) a x) ∂(κ + η) a = κ a s := by
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    s : Set γ
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral (((HAdd.hAdd κ η) a).restrict s) fun x => ENNRea …
  -/
  have h_le : κ ≤ κ + η := le_add_of_nonneg_right bot_le
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    s : Set γ
    hs : MeasurableSet s
    h_le : LE.le κ (HAdd.hAdd κ η)
    ⊢ Eq (MeasureTheory.lintegral (((HAdd.hAdd κ η) a).restrict s) fun x => ENNRea …
  -/
  simp_rw [rnDerivAux]
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    s : Set γ
    hs : MeasurableSet s
    h_le : LE.le κ (HAdd.hAdd κ η)
    ⊢ Eq (MeasureTheory.lintegral (((HAdd.hAdd κ η) a).restrict s) fun x => ENNRea …
  -/
  split_ifs with hα
    /-
      case pos
      α : Type u_1
      γ : Type u_2
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
      κ η : ProbabilityTheory.Kernel α γ
      inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
      inst✝ : ProbabilityTheory.IsFiniteKernel η
      a : α
      s : Set γ
      hs : MeasurableSet s
      h_le : LE.le κ (HAdd.hAdd κ η)
      hα : Countable α
      ⊢ Eq (MeasureTheory.lintegral (((HAdd.hAdd κ η) a).restrict s) fun x => ENNRea …
    -/
  · have h_ac : κ a ≪ (κ + η) a := Measure.absolutelyContinuous_of_le (h_le a)
    /-
      case pos
      α : Type u_1
      γ : Type u_2
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
      κ η : ProbabilityTheory.Kernel α γ
      inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
      inst✝ : ProbabilityTheory.IsFiniteKernel η
      a : α
      s : Set γ
      hs : MeasurableSet s
      h_le : LE.le κ (HAdd.hAdd κ η)
      hα : Countable α
      h_ac : (κ a).AbsolutelyContinuous ((HAdd.hAdd κ η) a)
      ⊢ Eq (MeasureTheory.lintegral (((HAdd.hAdd κ η) a).restrict s) fun x => ENNRea …
    -/
    rw [← Measure.setLIntegral_rnDeriv h_ac]
    /-
      case pos
      α : Type u_1
      γ : Type u_2
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
      κ η : ProbabilityTheory.Kernel α γ
      inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
      inst✝ : ProbabilityTheory.IsFiniteKernel η
      a : α
      s : Set γ
      hs : MeasurableSet s
      h_le : LE.le κ (HAdd.hAdd κ η)
      hα : Countable α
      h_ac : (κ a).AbsolutelyContinuous ((HAdd.hAdd κ η) a)
      ⊢ Eq (MeasureTheory.lintegral (((HAdd.hAdd κ η) a).restrict s) fun x => ENNRea …
    -/
    refine setLIntegral_congr_fun hs ?_
    /-
      case pos
      α : Type u_1
      γ : Type u_2
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
      κ η : ProbabilityTheory.Kernel α γ
      inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
      inst✝ : ProbabilityTheory.IsFiniteKernel η
      a : α
      s : Set γ
      hs : MeasurableSet s
      h_le : LE.le κ (HAdd.hAdd κ η)
      hα : Countable α
      h_ac : (κ a).AbsolutelyContinuous ((HAdd.hAdd κ η) a)
      ⊢ Filter.Eventually (fun x => Membership.mem s x → Eq (ENNReal.ofReal ((κ a).r …
    -/
    filter_upwards [Measure.rnDeriv_lt_top (κ a) ((κ + η) a)] with x hx_lt _
    /-
      case h
      α : Type u_1
      γ : Type u_2
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
      κ η : ProbabilityTheory.Kernel α γ
      inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
      inst✝ : ProbabilityTheory.IsFiniteKernel η
      a : α
      s : Set γ
      hs : MeasurableSet s
      h_le : LE.le κ (HAdd.hAdd κ η)
      hα : Countable α
      h_ac : (κ a).AbsolutelyContinuous ((HAdd.hAdd κ η) a)
      x : γ
      hx_lt : LT.lt ((κ a).rnDeriv ((HAdd.hAdd κ η) a) x) Top.top
      a✝ : Membership.mem s x
      ⊢ Eq (ENNReal.ofReal ((κ a).rnDeriv ((HAdd.hAdd κ η) a) x).toReal) ((κ a).rnDe …
    -/
    rw [ENNReal.ofReal_toReal hx_lt.ne]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      γ : Type u_2
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
      κ η : ProbabilityTheory.Kernel α γ
      inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
      inst✝ : ProbabilityTheory.IsFiniteKernel η
      a : α
      s : Set γ
      hs : MeasurableSet s
      h_le : LE.le κ (HAdd.hAdd κ η)
      hα : Not (Countable α)
      ⊢ Eq (MeasureTheory.lintegral (((HAdd.hAdd κ η) a).restrict s) fun x => ENNRea …
    -/
  · have := hαγ.countableOrCountablyGenerated.resolve_left hα
    rw [setLIntegral_density ((fst_map_id_prod _ measurable_const).trans_le h_le) _
      MeasurableSet.univ hs, map_apply' _ (by fun_prop) _ (hs.prod MeasurableSet.univ)]
    /-
      case neg
      α : Type u_1
      γ : Type u_2
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
      κ η : ProbabilityTheory.Kernel α γ
      inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
      inst✝ : ProbabilityTheory.IsFiniteKernel η
      a : α
      s : Set γ
      hs : MeasurableSet s
      h_le : LE.le κ (HAdd.hAdd κ η)
      hα : Not (Countable α)
      this : MeasurableSpace.CountablyGenerated γ
      ⊢ Eq ((κ a) (Set.preimage (fun a => { fst := a, snd := Unit.unit }) (SProd.spr …
    -/
    congr with x
    /-
      case neg.h.e_6.h.h
      α : Type u_1
      γ : Type u_2
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
      κ η : ProbabilityTheory.Kernel α γ
      inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
      inst✝ : ProbabilityTheory.IsFiniteKernel η
      a : α
      s : Set γ
      hs : MeasurableSet s
      h_le : LE.le κ (HAdd.hAdd κ η)
      hα : Not (Countable α)
      this : MeasurableSpace.CountablyGenerated γ
      x : γ
      ⊢ Iff (Membership.mem (Set.preimage (fun a => { fst := a, snd := Unit.unit })  …
    -/
    simp
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_rnDerivAux := setLIntegral_rnDerivAux


lemma withDensity_rnDerivAux (κ η : Kernel α γ) [IsFiniteKernel κ] [IsFiniteKernel η] :
    withDensity (κ + η) (fun a x ↦ Real.toNNReal (rnDerivAux κ (κ + η) a x)) = κ := by
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    ⊢ Eq ((HAdd.hAdd κ η).withDensity fun a x => ↑(κ.rnDerivAux (HAdd.hAdd κ η) a  …
  -/
  ext a s hs
  /-
    case h.h
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    s : Set γ
    hs : MeasurableSet s
    ⊢ Eq ((((HAdd.hAdd κ η).withDensity fun a x => ↑(κ.rnDerivAux (HAdd.hAdd κ η)  …
  -/
  rw [Kernel.withDensity_apply']
  /-
    case h.h
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    s : Set γ
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral (((HAdd.hAdd κ η) a).restrict s) fun b => ↑(κ.rn …
  -/
  swap
    /-
      case h.h.hf
      α : Type u_1
      γ : Type u_2
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
      κ η : ProbabilityTheory.Kernel α γ
      inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
      inst✝ : ProbabilityTheory.IsFiniteKernel η
      a : α
      s : Set γ
      hs : MeasurableSet s
      ⊢ Measurable (Function.uncurry fun a x => ↑(κ.rnDerivAux (HAdd.hAdd κ η) a x). …
    -/
  · exact (measurable_rnDerivAux _ _).ennreal_ofReal
    /-
      🎉 no goals
    -/
  /-
    case h.h
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    s : Set γ
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral (((HAdd.hAdd κ η) a).restrict s) fun b => ↑(κ.rn …
  -/
  simp_rw [ofNNReal_toNNReal]
  /-
    case h.h
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    s : Set γ
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral (((HAdd.hAdd κ η) a).restrict s) fun b => ENNRea …
  -/
  exact setLIntegral_rnDerivAux κ η a hs
  /-
    🎉 no goals
  -/


lemma withDensity_one_sub_rnDerivAux (κ η : Kernel α γ) [IsFiniteKernel κ] [IsFiniteKernel η] :
    withDensity (κ + η) (fun a x ↦ Real.toNNReal (1 - rnDerivAux κ (κ + η) a x)) = η := by
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    ⊢ Eq ((HAdd.hAdd κ η).withDensity fun a x => ↑(HSub.hSub 1 (κ.rnDerivAux (HAdd …
  -/
  have h_le : κ ≤ κ + η := le_add_of_nonneg_right bot_le
  suffices withDensity (κ + η) (fun a x ↦ Real.toNNReal (1 - rnDerivAux κ (κ + η) a x))
      + withDensity (κ + η) (fun a x ↦ Real.toNNReal (rnDerivAux κ (κ + η) a x))
      = κ + η by
    ext a s
    have h : (withDensity (κ + η) (fun a x ↦ Real.toNNReal (1 - rnDerivAux κ (κ + η) a x))
          + withDensity (κ + η) (fun a x ↦ Real.toNNReal (rnDerivAux κ (κ + η) a x))) a s
        = κ a s + η a s := by
      rw [this]
      simp
    simp only [coe_add, Pi.add_apply, Measure.coe_add] at h
    rwa [withDensity_rnDerivAux, add_comm, ENNReal.add_right_inj (measure_ne_top _ _)] at h
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    h_le : LE.le κ (HAdd.hAdd κ η)
    ⊢ Eq (HAdd.hAdd ((HAdd.hAdd κ η).withDensity fun a x => ↑(HSub.hSub 1 (κ.rnDer …
  -/
  have : ∀ b, (Real.toNNReal b : ℝ≥0∞) = ENNReal.ofReal b := fun _ ↦ rfl
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    h_le : LE.le κ (HAdd.hAdd κ η)
    this : ∀ (b : Real), Eq (↑b.toNNReal) (ENNReal.ofReal b)
    ⊢ Eq (HAdd.hAdd ((HAdd.hAdd κ η).withDensity fun a x => ↑(HSub.hSub 1 (κ.rnDer …
  -/
  simp_rw [this, ENNReal.ofReal_sub _ (rnDerivAux_nonneg h_le), ENNReal.ofReal_one]
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    h_le : LE.le κ (HAdd.hAdd κ η)
    this : ∀ (b : Real), Eq (↑b.toNNReal) (ENNReal.ofReal b)
    ⊢ Eq (HAdd.hAdd ((HAdd.hAdd κ η).withDensity fun a x => HSub.hSub 1 (ENNReal.o …
  -/
  rw [withDensity_sub_add_cancel]
    /-
      α : Type u_1
      γ : Type u_2
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
      κ η : ProbabilityTheory.Kernel α γ
      inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
      inst✝ : ProbabilityTheory.IsFiniteKernel η
      h_le : LE.le κ (HAdd.hAdd κ η)
      this : ∀ (b : Real), Eq (↑b.toNNReal) (ENNReal.ofReal b)
      ⊢ Eq ((HAdd.hAdd κ η).withDensity fun a x => 1) (HAdd.hAdd κ η)
    -/
  · rw [withDensity_one']
    /-
      🎉 no goals
    -/
    /-
      case hf
      α : Type u_1
      γ : Type u_2
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
      κ η : ProbabilityTheory.Kernel α γ
      inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
      inst✝ : ProbabilityTheory.IsFiniteKernel η
      h_le : LE.le κ (HAdd.hAdd κ η)
      this : ∀ (b : Real), Eq (↑b.toNNReal) (ENNReal.ofReal b)
      ⊢ Measurable (Function.uncurry fun a x => 1)
    -/
  · exact measurable_const
    /-
      🎉 no goals
    -/
    /-
      case hg
      α : Type u_1
      γ : Type u_2
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
      κ η : ProbabilityTheory.Kernel α γ
      inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
      inst✝ : ProbabilityTheory.IsFiniteKernel η
      h_le : LE.le κ (HAdd.hAdd κ η)
      this : ∀ (b : Real), Eq (↑b.toNNReal) (ENNReal.ofReal b)
      ⊢ Measurable (Function.uncurry fun a x => ENNReal.ofReal (κ.rnDerivAux (HAdd.h …
    -/
  · exact (measurable_rnDerivAux _ _).ennreal_ofReal
    /-
      🎉 no goals
    -/
    /-
      case hfg
      α : Type u_1
      γ : Type u_2
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
      κ η : ProbabilityTheory.Kernel α γ
      inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
      inst✝ : ProbabilityTheory.IsFiniteKernel η
      h_le : LE.le κ (HAdd.hAdd κ η)
      this : ∀ (b : Real), Eq (↑b.toNNReal) (ENNReal.ofReal b)
      ⊢ ∀ (a : α), (MeasureTheory.ae ((HAdd.hAdd κ η) a)).EventuallyLE (fun x => ENN …
    -/
  · intro a
    /-
      case hfg
      α : Type u_1
      γ : Type u_2
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
      κ η : ProbabilityTheory.Kernel α γ
      inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
      inst✝ : ProbabilityTheory.IsFiniteKernel η
      h_le : LE.le κ (HAdd.hAdd κ η)
      this : ∀ (b : Real), Eq (↑b.toNNReal) (ENNReal.ofReal b)
      a : α
      ⊢ (MeasureTheory.ae ((HAdd.hAdd κ η) a)).EventuallyLE (fun x => ENNReal.ofReal …
    -/
    filter_upwards [rnDerivAux_le_one h_le] with x hx
    /-
      case h
      α : Type u_1
      γ : Type u_2
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
      κ η : ProbabilityTheory.Kernel α γ
      inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
      inst✝ : ProbabilityTheory.IsFiniteKernel η
      h_le : LE.le κ (HAdd.hAdd κ η)
      this : ∀ (b : Real), Eq (↑b.toNNReal) (ENNReal.ofReal b)
      a : α
      x : γ
      hx : LE.le (κ.rnDerivAux (HAdd.hAdd κ η) a x) (1 x)
      ⊢ LE.le (ENNReal.ofReal (κ.rnDerivAux (HAdd.hAdd κ η) a x)) 1
    -/
    simp only [ENNReal.ofReal_le_one]
    /-
      case h
      α : Type u_1
      γ : Type u_2
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
      κ η : ProbabilityTheory.Kernel α γ
      inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
      inst✝ : ProbabilityTheory.IsFiniteKernel η
      h_le : LE.le κ (HAdd.hAdd κ η)
      this : ∀ (b : Real), Eq (↑b.toNNReal) (ENNReal.ofReal b)
      a : α
      x : γ
      hx : LE.le (κ.rnDerivAux (HAdd.hAdd κ η) a x) (1 x)
      ⊢ LE.le (κ.rnDerivAux (HAdd.hAdd κ η) a x) 1
    -/
    exact hx
    /-
      🎉 no goals
    -/


/-- A set of points in `α × γ` related to the absolute continuity / mutual singularity of
`κ` and `η`. -/
def mutuallySingularSet (κ η : Kernel α γ) : Set (α × γ) := {p | 1 ≤ rnDerivAux κ (κ + η) p.1 p.2}


/-- A set of points in `α × γ` related to the absolute continuity / mutual singularity of
`κ` and `η`. That is,
* `withDensity η (rnDeriv κ η) a (mutuallySingularSetSlice κ η a) = 0`,
* `singularPart κ η a (mutuallySingularSetSlice κ η a)ᶜ = 0`.
 -/
def mutuallySingularSetSlice (κ η : Kernel α γ) (a : α) : Set γ :=
  {x | 1 ≤ rnDerivAux κ (κ + η) a x}


lemma mem_mutuallySingularSetSlice (κ η : Kernel α γ) (a : α) (x : γ) :
    x ∈ mutuallySingularSetSlice κ η a ↔ 1 ≤ rnDerivAux κ (κ + η) a x := by
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    a : α
    x : γ
    ⊢ Iff (Membership.mem (κ.mutuallySingularSetSlice η a) x) (LE.le 1 (κ.rnDerivA …
  -/
  rw [mutuallySingularSetSlice]; rfl
                                 /-
                                   🎉 no goals
                                 -/


lemma not_mem_mutuallySingularSetSlice (κ η : Kernel α γ) (a : α) (x : γ) :
    x ∉ mutuallySingularSetSlice κ η a ↔ rnDerivAux κ (κ + η) a x < 1 := by
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    a : α
    x : γ
    ⊢ Iff (Not (Membership.mem (κ.mutuallySingularSetSlice η a) x)) (LT.lt (κ.rnDe …
  -/
  simp [mutuallySingularSetSlice]
  /-
    🎉 no goals
  -/


lemma measurableSet_mutuallySingularSet (κ η : Kernel α γ) :
    MeasurableSet (mutuallySingularSet κ η) :=
  measurable_rnDerivAux κ (κ + η) measurableSet_Ici


lemma measurableSet_mutuallySingularSetSlice (κ η : Kernel α γ) (a : α) :
    MeasurableSet (mutuallySingularSetSlice κ η a) :=
  measurable_prod_mk_left (measurableSet_mutuallySingularSet κ η)


lemma measure_mutuallySingularSetSlice (κ η : Kernel α γ) [IsFiniteKernel κ] [IsFiniteKernel η]
    (a : α) :
    η a (mutuallySingularSetSlice κ η a) = 0 := by
  suffices withDensity (κ + η) (fun a x ↦ Real.toNNReal
      (1 - rnDerivAux κ (κ + η) a x)) a {x | 1 ≤ rnDerivAux κ (κ + η) a x} = 0 by
    rwa [withDensity_one_sub_rnDerivAux κ η] at this
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    ⊢ Eq ((((HAdd.hAdd κ η).withDensity fun a x => ↑(HSub.hSub 1 (κ.rnDerivAux (HA …
  -/
  simp_rw [ofNNReal_toNNReal]
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    ⊢ Eq ((((HAdd.hAdd κ η).withDensity fun a x => ENNReal.ofReal (HSub.hSub 1 (κ. …
  -/
  rw [Kernel.withDensity_apply', lintegral_eq_zero_iff, EventuallyEq, ae_restrict_iff]
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    ⊢ Filter.Eventually (fun x => Membership.mem (setOf fun x => LE.le 1 (κ.rnDeri …
  -/
  rotate_left
  · exact (measurable_const.sub
      ((measurable_rnDerivAux _ _).comp measurable_prod_mk_left)).ennreal_ofReal
      (measurableSet_singleton _)
  · exact (measurable_const.sub
      ((measurable_rnDerivAux _ _).comp measurable_prod_mk_left)).ennreal_ofReal
    /-
      case hf
      α : Type u_1
      γ : Type u_2
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
      κ η : ProbabilityTheory.Kernel α γ
      inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
      inst✝ : ProbabilityTheory.IsFiniteKernel η
      a : α
      ⊢ Measurable (Function.uncurry fun a x => ENNReal.ofReal (HSub.hSub 1 (κ.rnDer …
    -/
  · exact (measurable_const.sub (measurable_rnDerivAux _ _)).ennreal_ofReal
    /-
      🎉 no goals
    -/
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    ⊢ Filter.Eventually (fun x => Membership.mem (setOf fun x => LE.le 1 (κ.rnDeri …
  -/
  refine ae_of_all _ (fun x hx ↦ ?_)
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    x : γ
    hx : Membership.mem (setOf fun x => LE.le 1 (κ.rnDerivAux (HAdd.hAdd κ η) a x) …
    ⊢ Eq (ENNReal.ofReal (HSub.hSub 1 (κ.rnDerivAux (HAdd.hAdd κ η) a x))) (0 x)
  -/
  simp only [mem_setOf_eq] at hx
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    x : γ
    hx : LE.le 1 (κ.rnDerivAux (HAdd.hAdd κ η) a x)
    ⊢ Eq (ENNReal.ofReal (HSub.hSub 1 (κ.rnDerivAux (HAdd.hAdd κ η) a x))) (0 x)
  -/
  simp [hx]
  /-
    🎉 no goals
  -/


/-- Radon-Nikodym derivative of the kernel `κ` with respect to the kernel `η`. -/
noncomputable
irreducible_def rnDeriv (κ η : Kernel α γ) (a : α) (x : γ) : ℝ≥0∞ :=
  ENNReal.ofReal (rnDerivAux κ (κ + η) a x) / ENNReal.ofReal (1 - rnDerivAux κ (κ + η) a x)


lemma rnDeriv_def' (κ η : Kernel α γ) :
    rnDeriv κ η = fun a x ↦ ENNReal.ofReal (rnDerivAux κ (κ + η) a x)
                                                            /-
                                                              α : Type u_1
                                                              γ : Type u_2
                                                              mα : MeasurableSpace α
                                                              mγ : MeasurableSpace γ
                                                              hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
                                                              κ η : ProbabilityTheory.Kernel α γ
                                                              ⊢ Eq (κ.rnDeriv η) fun a x => HDiv.hDiv (ENNReal.ofReal (κ.rnDerivAux (HAdd.hA …
                                                            -/
      / ENNReal.ofReal (1 - rnDerivAux κ (κ + η) a x) := by ext; rw [rnDeriv_def]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


lemma measurable_rnDeriv (κ η : Kernel α γ) :
    Measurable (fun p : α × γ ↦ rnDeriv κ η p.1 p.2) := by
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    ⊢ Measurable fun p => κ.rnDeriv η p.1 p.2
  -/
  simp_rw [rnDeriv_def]
  exact (measurable_rnDerivAux κ _).ennreal_ofReal.div
    (measurable_const.sub (measurable_rnDerivAux κ _)).ennreal_ofReal


lemma measurable_rnDeriv_right (κ η : Kernel α γ) (a : α) :
    Measurable (fun x : γ ↦ rnDeriv κ η a x) := by
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    a : α
    ⊢ Measurable fun x => κ.rnDeriv η a x
  -/
  change Measurable ((fun p : α × γ ↦ rnDeriv κ η p.1 p.2) ∘ (fun x ↦ (a, x)))
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    a : α
    ⊢ Measurable (Function.comp (fun p => κ.rnDeriv η p.1 p.2) fun x => { fst := a …
  -/
  exact (measurable_rnDeriv _ _).comp measurable_prod_mk_left
  /-
    🎉 no goals
  -/


lemma rnDeriv_eq_top_iff (κ η : Kernel α γ) (a : α) (x : γ) :
    rnDeriv κ η a x = ∞ ↔ (a, x) ∈ mutuallySingularSet κ η := by
  simp only [rnDeriv, ENNReal.div_eq_top, ne_eq, ENNReal.ofReal_eq_zero, not_le,
    tsub_le_iff_right, zero_add, ENNReal.ofReal_ne_top, not_false_eq_true, and_true, or_false,
    mutuallySingularSet, mem_setOf_eq, and_iff_right_iff_imp]
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    a : α
    x : γ
    ⊢ LE.le 1 (κ.rnDerivAux (HAdd.hAdd κ η) a x) → LT.lt 0 (κ.rnDerivAux (HAdd.hAd …
  -/
  exact fun h ↦ zero_lt_one.trans_le h
  /-
    🎉 no goals
  -/


lemma rnDeriv_eq_top_iff' (κ η : Kernel α γ) (a : α) (x : γ) :
    rnDeriv κ η a x = ∞ ↔ x ∈ mutuallySingularSetSlice κ η a := by
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    a : α
    x : γ
    ⊢ Iff (Eq (κ.rnDeriv η a x) Top.top) (Membership.mem (κ.mutuallySingularSetSli …
  -/
  rw [rnDeriv_eq_top_iff, mutuallySingularSet, mutuallySingularSetSlice, mem_setOf, mem_setOf]
  /-
    🎉 no goals
  -/


/-- Singular part of the kernel `κ` with respect to the kernel `η`. -/
noncomputable
irreducible_def singularPart (κ η : Kernel α γ) [IsSFiniteKernel κ] [IsSFiniteKernel η] :
    Kernel α γ :=
  withDensity (κ + η) (fun a x ↦ Real.toNNReal (rnDerivAux κ (κ + η) a x)
    - Real.toNNReal (1 - rnDerivAux κ (κ + η) a x) * rnDeriv κ η a x)


lemma measurable_singularPart_fun (κ η : Kernel α γ) :
    Measurable (fun p : α × γ ↦ Real.toNNReal (rnDerivAux κ (κ + η) p.1 p.2)
      - Real.toNNReal (1 - rnDerivAux κ (κ + η) p.1 p.2) * rnDeriv κ η p.1 p.2) :=
  (measurable_rnDerivAux _ _).ennreal_ofReal.sub
    ((measurable_const.sub (measurable_rnDerivAux _ _)).ennreal_ofReal.mul (measurable_rnDeriv _ _))


lemma measurable_singularPart_fun_right (κ η : Kernel α γ) (a : α) :
    Measurable (fun x : γ ↦ Real.toNNReal (rnDerivAux κ (κ + η) a x)
      - Real.toNNReal (1 - rnDerivAux κ (κ + η) a x) * rnDeriv κ η a x) := by
  change Measurable ((Function.uncurry fun a b ↦
    ENNReal.ofReal (rnDerivAux κ (κ + η) a b)
    - ENNReal.ofReal (1 - rnDerivAux κ (κ + η) a b) * rnDeriv κ η a b) ∘ (fun b ↦ (a, b)))
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    a : α
    ⊢ Measurable (Function.comp (Function.uncurry fun a b => HSub.hSub (ENNReal.of …
  -/
  exact (measurable_singularPart_fun κ η).comp measurable_prod_mk_left
  /-
    🎉 no goals
  -/


lemma singularPart_compl_mutuallySingularSetSlice (κ η : Kernel α γ) [IsSFiniteKernel κ]
    [IsSFiniteKernel η] (a : α) :
    singularPart κ η a (mutuallySingularSetSlice κ η a)ᶜ = 0 := by
  rw [singularPart, Kernel.withDensity_apply', lintegral_eq_zero_iff, EventuallyEq,
    ae_restrict_iff]
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    ⊢ Filter.Eventually (fun x => Membership.mem (HasCompl.compl (κ.mutuallySingul …
  -/
  all_goals simp_rw [ofNNReal_toNNReal]
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    ⊢ Filter.Eventually (fun x => Membership.mem (HasCompl.compl (κ.mutuallySingul …
  -/
  rotate_left
  · exact measurableSet_preimage (measurable_singularPart_fun_right κ η a)
      (measurableSet_singleton _)
    /-
      α : Type u_1
      γ : Type u_2
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
      κ η : ProbabilityTheory.Kernel α γ
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      ⊢ Measurable fun b => HSub.hSub (ENNReal.ofReal (κ.rnDerivAux (HAdd.hAdd κ η)  …
    -/
  · exact measurable_singularPart_fun_right κ η a
    /-
      🎉 no goals
    -/
    /-
      case hf
      α : Type u_1
      γ : Type u_2
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
      κ η : ProbabilityTheory.Kernel α γ
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      ⊢ Measurable (Function.uncurry fun a x => HSub.hSub (ENNReal.ofReal (κ.rnDeriv …
    -/
  · exact measurable_singularPart_fun κ η
    /-
      🎉 no goals
    -/
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    ⊢ Filter.Eventually (fun x => Membership.mem (HasCompl.compl (κ.mutuallySingul …
  -/
  refine ae_of_all _ (fun x hx ↦ ?_)
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    x : γ
    hx : Membership.mem (HasCompl.compl (κ.mutuallySingularSetSlice η a)) x
    ⊢ Eq (HSub.hSub (ENNReal.ofReal (κ.rnDerivAux (HAdd.hAdd κ η) a x)) (HMul.hMul …
  -/
  simp only [mem_compl_iff, mutuallySingularSetSlice, mem_setOf, not_le] at hx
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    x : γ
    hx : LT.lt (κ.rnDerivAux (HAdd.hAdd κ η) a x) 1
    ⊢ Eq (HSub.hSub (ENNReal.ofReal (κ.rnDerivAux (HAdd.hAdd κ η) a x)) (HMul.hMul …
  -/
  simp_rw [rnDeriv]
  rw [← ENNReal.ofReal_div_of_pos, div_eq_inv_mul, ← ENNReal.ofReal_mul, ← mul_assoc,
    mul_inv_cancel₀, one_mul, tsub_self, Pi.zero_apply]
    /-
      α : Type u_1
      γ : Type u_2
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
      κ η : ProbabilityTheory.Kernel α γ
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      x : γ
      hx : LT.lt (κ.rnDerivAux (HAdd.hAdd κ η) a x) 1
      ⊢ Ne (HSub.hSub 1 (κ.rnDerivAux (HAdd.hAdd κ η) a x)) 0
    -/
  · simp only [ne_eq, sub_eq_zero, hx.ne', not_false_eq_true]
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      γ : Type u_2
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
      κ η : ProbabilityTheory.Kernel α γ
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      x : γ
      hx : LT.lt (κ.rnDerivAux (HAdd.hAdd κ η) a x) 1
      ⊢ LE.le 0 (HSub.hSub 1 (κ.rnDerivAux (HAdd.hAdd κ η) a x))
    -/
  · simp only [sub_nonneg, hx.le]
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      γ : Type u_2
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
      κ η : ProbabilityTheory.Kernel α γ
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      x : γ
      hx : LT.lt (κ.rnDerivAux (HAdd.hAdd κ η) a x) 1
      ⊢ LT.lt 0 (HSub.hSub 1 (κ.rnDerivAux (HAdd.hAdd κ η) a x))
    -/
  · simp only [sub_pos, hx]
    /-
      🎉 no goals
    -/


lemma singularPart_of_subset_compl_mutuallySingularSetSlice [IsFiniteKernel κ]
    [IsFiniteKernel η] {a : α} {s : Set γ} (hs : s ⊆ (mutuallySingularSetSlice κ η a)ᶜ) :
    singularPart κ η a s = 0 :=
  measure_mono_null hs (singularPart_compl_mutuallySingularSetSlice κ η a)


lemma singularPart_of_subset_mutuallySingularSetSlice [IsFiniteKernel κ]
    [IsFiniteKernel η] {a : α} {s : Set γ} (hsm : MeasurableSet s)
    (hs : s ⊆ mutuallySingularSetSlice κ η a) :
    singularPart κ η a s = κ a s := by
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    κ η : ProbabilityTheory.Kernel α γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    s : Set γ
    hsm : MeasurableSet s
    hs : HasSubset.Subset s (κ.mutuallySingularSetSlice η a)
    ⊢ Eq (((κ.singularPart η) a) s) ((κ a) s)
  -/
  have hs' : ∀ x ∈ s, 1 ≤ rnDerivAux κ (κ + η) a x := fun _ hx ↦ hs hx
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    κ η : ProbabilityTheory.Kernel α γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    s : Set γ
    hsm : MeasurableSet s
    hs : HasSubset.Subset s (κ.mutuallySingularSetSlice η a)
    hs' : ∀ (x : γ), Membership.mem s x → LE.le 1 (κ.rnDerivAux (HAdd.hAdd κ η) a x)
    ⊢ Eq (((κ.singularPart η) a) s) ((κ a) s)
  -/
  rw [singularPart, Kernel.withDensity_apply']
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    κ η : ProbabilityTheory.Kernel α γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    s : Set γ
    hsm : MeasurableSet s
    hs : HasSubset.Subset s (κ.mutuallySingularSetSlice η a)
    hs' : ∀ (x : γ), Membership.mem s x → LE.le 1 (κ.rnDerivAux (HAdd.hAdd κ η) a x)
    ⊢ Eq (MeasureTheory.lintegral (((HAdd.hAdd κ η) a).restrict s) fun b => HSub.h …
  -/
  swap; · exact measurable_singularPart_fun κ η
          /-
            🎉 no goals
          -/
  calc
    ∫⁻ x in s, ↑(Real.toNNReal (rnDerivAux κ (κ + η) a x)) -
      ↑(Real.toNNReal (1 - rnDerivAux κ (κ + η) a x)) * rnDeriv κ η a x
      ∂(κ + η) a
    = ∫⁻ _ in s, 1 ∂(κ + η) a := by
        refine setLIntegral_congr_fun hsm ?_
        have h_le : κ ≤ κ + η := le_add_of_nonneg_right bot_le
        filter_upwards [rnDerivAux_le_one h_le] with x hx hxs
        have h_eq_one : rnDerivAux κ (κ + η) a x = 1 := le_antisymm hx (hs' x hxs)
        simp [h_eq_one]
  _ = (κ + η) a s := by simp
  _ = κ a s := by
        suffices η a s = 0 by simp [this]
        exact measure_mono_null hs (measure_mutuallySingularSetSlice κ η a)


lemma withDensity_rnDeriv_mutuallySingularSetSlice (κ η : Kernel α γ) [IsFiniteKernel κ]
    [IsFiniteKernel η] (a : α) :
    withDensity η (rnDeriv κ η) a (mutuallySingularSetSlice κ η a) = 0 := by
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    ⊢ Eq (((η.withDensity (κ.rnDeriv η)) a) (κ.mutuallySingularSetSlice η a)) 0
  -/
  rw [Kernel.withDensity_apply']
    /-
      α : Type u_1
      γ : Type u_2
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
      κ η : ProbabilityTheory.Kernel α γ
      inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
      inst✝ : ProbabilityTheory.IsFiniteKernel η
      a : α
      ⊢ Eq (MeasureTheory.lintegral ((η a).restrict (κ.mutuallySingularSetSlice η a) …
    -/
  · exact setLIntegral_measure_zero _ _ (measure_mutuallySingularSetSlice κ η a)
    /-
      🎉 no goals
    -/
    /-
      case hf
      α : Type u_1
      γ : Type u_2
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
      κ η : ProbabilityTheory.Kernel α γ
      inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
      inst✝ : ProbabilityTheory.IsFiniteKernel η
      a : α
      ⊢ Measurable (Function.uncurry (κ.rnDeriv η))
    -/
  · exact measurable_rnDeriv κ η
    /-
      🎉 no goals
    -/


lemma withDensity_rnDeriv_of_subset_mutuallySingularSetSlice [IsFiniteKernel κ]
    [IsFiniteKernel η] {a : α} {s : Set γ}
    (hs : s ⊆ mutuallySingularSetSlice κ η a) :
    withDensity η (rnDeriv κ η) a s = 0 :=
  measure_mono_null hs (withDensity_rnDeriv_mutuallySingularSetSlice κ η a)


lemma withDensity_rnDeriv_of_subset_compl_mutuallySingularSetSlice
    [IsFiniteKernel κ] [IsFiniteKernel η] {a : α} {s : Set γ} (hsm : MeasurableSet s)
    (hs : s ⊆ (mutuallySingularSetSlice κ η a)ᶜ) :
    withDensity η (rnDeriv κ η) a s = κ a s := by
  have : withDensity η (rnDeriv κ η)
      = withDensity (withDensity (κ + η)
        (fun a x ↦ Real.toNNReal (1 - rnDerivAux κ (κ + η) a x))) (rnDeriv κ η) := by
    rw [rnDeriv_def']
    congr
    exact (withDensity_one_sub_rnDerivAux κ η).symm
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    κ η : ProbabilityTheory.Kernel α γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    s : Set γ
    hsm : MeasurableSet s
    hs : HasSubset.Subset s (HasCompl.compl (κ.mutuallySingularSetSlice η a))
    this : Eq (η.withDensity (κ.rnDeriv η)) (((HAdd.hAdd κ η).withDensity fun a x  …
    ⊢ Eq (((η.withDensity (κ.rnDeriv η)) a) s) ((κ a) s)
  -/
  rw [this, ← withDensity_mul, Kernel.withDensity_apply']
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    κ η : ProbabilityTheory.Kernel α γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    s : Set γ
    hsm : MeasurableSet s
    hs : HasSubset.Subset s (HasCompl.compl (κ.mutuallySingularSetSlice η a))
    this : Eq (η.withDensity (κ.rnDeriv η)) (((HAdd.hAdd κ η).withDensity fun a x  …
    ⊢ Eq (MeasureTheory.lintegral (((HAdd.hAdd κ η) a).restrict s) fun b => HMul.h …
  -/
  rotate_left
  · exact ((measurable_const.sub (measurable_rnDerivAux _ _)).ennreal_ofReal.mul
    (measurable_rnDeriv _ _))
    /-
      case hf
      α : Type u_1
      γ : Type u_2
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      κ η : ProbabilityTheory.Kernel α γ
      hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
      inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
      inst✝ : ProbabilityTheory.IsFiniteKernel η
      a : α
      s : Set γ
      hsm : MeasurableSet s
      hs : HasSubset.Subset s (HasCompl.compl (κ.mutuallySingularSetSlice η a))
      this : Eq (η.withDensity (κ.rnDeriv η)) (((HAdd.hAdd κ η).withDensity fun a x  …
      ⊢ Measurable (Function.uncurry fun a x => (HSub.hSub 1 (κ.rnDerivAux (HAdd.hAd …
    -/
  · exact (measurable_const.sub (measurable_rnDerivAux _ _)).real_toNNReal
    /-
      🎉 no goals
    -/
    /-
      case hg
      α : Type u_1
      γ : Type u_2
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      κ η : ProbabilityTheory.Kernel α γ
      hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
      inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
      inst✝ : ProbabilityTheory.IsFiniteKernel η
      a : α
      s : Set γ
      hsm : MeasurableSet s
      hs : HasSubset.Subset s (HasCompl.compl (κ.mutuallySingularSetSlice η a))
      this : Eq (η.withDensity (κ.rnDeriv η)) (((HAdd.hAdd κ η).withDensity fun a x  …
      ⊢ Measurable (Function.uncurry (κ.rnDeriv η))
    -/
  · exact measurable_rnDeriv _ _
    /-
      🎉 no goals
    -/
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    κ η : ProbabilityTheory.Kernel α γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    s : Set γ
    hsm : MeasurableSet s
    hs : HasSubset.Subset s (HasCompl.compl (κ.mutuallySingularSetSlice η a))
    this : Eq (η.withDensity (κ.rnDeriv η)) (((HAdd.hAdd κ η).withDensity fun a x  …
    ⊢ Eq (MeasureTheory.lintegral (((HAdd.hAdd κ η) a).restrict s) fun b => HMul.h …
  -/
  simp_rw [rnDeriv]
  have hs' : ∀ x ∈ s, rnDerivAux κ (κ + η) a x < 1 := by
    simp_rw [← not_mem_mutuallySingularSetSlice]
    exact fun x hx hx_mem ↦ hs hx hx_mem
  calc
    ∫⁻ x in s, ↑(Real.toNNReal (1 - rnDerivAux κ (κ + η) a x)) *
      (ENNReal.ofReal (rnDerivAux κ (κ + η) a x) /
        ENNReal.ofReal (1 - rnDerivAux κ (κ + η) a x)) ∂(κ + η) a
  _ = ∫⁻ x in s, ENNReal.ofReal (rnDerivAux κ (κ + η) a x) ∂(κ + η) a := by
      refine setLIntegral_congr_fun hsm (ae_of_all _ fun x hx ↦ ?_)
      rw [ofNNReal_toNNReal, ← ENNReal.ofReal_div_of_pos, div_eq_inv_mul, ← ENNReal.ofReal_mul,
        ← mul_assoc, mul_inv_cancel₀, one_mul]
      · rw [ne_eq, sub_eq_zero]
        exact (hs' x hx).ne'
      · simp [(hs' x hx).le]
      · simp [hs' x hx]
  _ = κ a s := setLIntegral_rnDerivAux κ η a hsm


/-- The singular part of `κ` with respect to `η` is mutually singular with `η`. -/
lemma mutuallySingular_singularPart (κ η : Kernel α γ) [IsFiniteKernel κ] [IsFiniteKernel η]
    (a : α) :
    singularPart κ η a ⟂ₘ η a := by
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    ⊢ ((κ.singularPart η) a).MutuallySingular (η a)
  -/
  symm
  exact ⟨mutuallySingularSetSlice κ η a, measurableSet_mutuallySingularSetSlice κ η a,
    measure_mutuallySingularSetSlice κ η a, singularPart_compl_mutuallySingularSetSlice κ η a⟩


/-- Lebesgue decomposition of a finite kernel `κ` with respect to another one `η`.
`κ` is the sum of an absolutely continuous part `withDensity η (rnDeriv κ η)` and a singular part
`singularPart κ η`. -/
lemma rnDeriv_add_singularPart (κ η : Kernel α γ) [IsFiniteKernel κ] [IsFiniteKernel η] :
    withDensity η (rnDeriv κ η) + singularPart κ η = κ := by
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    ⊢ Eq (HAdd.hAdd (η.withDensity (κ.rnDeriv η)) (κ.singularPart η)) κ
  -/
  ext a s hs
  /-
    case h.h
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    s : Set γ
    hs : MeasurableSet s
    ⊢ Eq (((HAdd.hAdd (η.withDensity (κ.rnDeriv η)) (κ.singularPart η)) a) s) ((κ  …
  -/
  rw [← inter_union_diff s (mutuallySingularSetSlice κ η a)]
  /-
    case h.h
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    s : Set γ
    hs : MeasurableSet s
    ⊢ Eq (((HAdd.hAdd (η.withDensity (κ.rnDeriv η)) (κ.singularPart η)) a) (Union. …
  -/
  simp only [coe_add, Pi.add_apply, Measure.coe_add]
  /-
    case h.h
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    s : Set γ
    hs : MeasurableSet s
    ⊢ Eq (HAdd.hAdd (((η.withDensity (κ.rnDeriv η)) a) (Union.union (Inter.inter s …
  -/
  have hm := measurableSet_mutuallySingularSetSlice κ η a
  simp only [measure_union (Disjoint.mono inter_subset_right le_rfl disjoint_sdiff_right)
    (hs.diff hm)]
  rw [singularPart_of_subset_mutuallySingularSetSlice (hs.inter hm) inter_subset_right,
    singularPart_of_subset_compl_mutuallySingularSetSlice (diff_subset_iff.mpr (by simp)),
    add_zero, withDensity_rnDeriv_of_subset_mutuallySingularSetSlice inter_subset_right,
    zero_add, withDensity_rnDeriv_of_subset_compl_mutuallySingularSetSlice (hs.diff hm)
      (diff_subset_iff.mpr (by simp)), add_comm]


lemma singularPart_eq_zero_iff_apply_eq_zero (κ η : Kernel α γ) [IsFiniteKernel κ]
    [IsFiniteKernel η] (a : α) :
    singularPart κ η a = 0 ↔ singularPart κ η a (mutuallySingularSetSlice κ η a) = 0 := by
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    ⊢ Iff (Eq ((κ.singularPart η) a) 0) (Eq (((κ.singularPart η) a) (κ.mutuallySin …
  -/
  rw [← Measure.measure_univ_eq_zero]
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    ⊢ Iff (Eq (((κ.singularPart η) a) Set.univ) 0) (Eq (((κ.singularPart η) a) (κ. …
  -/
  have : univ = (mutuallySingularSetSlice κ η a) ∪ (mutuallySingularSetSlice κ η a)ᶜ := by simp
  rw [this, measure_union disjoint_compl_right (measurableSet_mutuallySingularSetSlice κ η a).compl,
    singularPart_compl_mutuallySingularSetSlice, add_zero]


lemma withDensity_rnDeriv_eq_zero_iff_apply_eq_zero (κ η : Kernel α γ) [IsFiniteKernel κ]
    [IsFiniteKernel η] (a : α) :
    withDensity η (rnDeriv κ η) a = 0
      ↔ withDensity η (rnDeriv κ η) a (mutuallySingularSetSlice κ η a)ᶜ = 0 := by
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    ⊢ Iff (Eq ((η.withDensity (κ.rnDeriv η)) a) 0) (Eq (((η.withDensity (κ.rnDeriv …
  -/
  rw [← Measure.measure_univ_eq_zero]
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    ⊢ Iff (Eq (((η.withDensity (κ.rnDeriv η)) a) Set.univ) 0) (Eq (((η.withDensity …
  -/
  have : univ = (mutuallySingularSetSlice κ η a) ∪ (mutuallySingularSetSlice κ η a)ᶜ := by simp
  rw [this, measure_union disjoint_compl_right (measurableSet_mutuallySingularSetSlice κ η a).compl,
    withDensity_rnDeriv_mutuallySingularSetSlice, zero_add]


lemma singularPart_eq_zero_iff_absolutelyContinuous (κ η : Kernel α γ)
    [IsFiniteKernel κ] [IsFiniteKernel η] (a : α) :
    singularPart κ η a = 0 ↔ κ a ≪ η a := by
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    ⊢ Iff (Eq ((κ.singularPart η) a) 0) ((κ a).AbsolutelyContinuous (η a))
  -/
  conv_rhs => rw [← rnDeriv_add_singularPart κ η, coe_add, Pi.add_apply]
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    ⊢ Iff (Eq ((κ.singularPart η) a) 0) ((HAdd.hAdd ((η.withDensity (κ.rnDeriv η)) …
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ ?_⟩
    /-
      case refine_1
      α : Type u_1
      γ : Type u_2
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
      κ η : ProbabilityTheory.Kernel α γ
      inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
      inst✝ : ProbabilityTheory.IsFiniteKernel η
      a : α
      h : Eq ((κ.singularPart η) a) 0
      ⊢ (HAdd.hAdd ((η.withDensity (κ.rnDeriv η)) a) ((κ.singularPart η) a)).Absolut …
    -/
  · rw [h, add_zero]
    /-
      case refine_1
      α : Type u_1
      γ : Type u_2
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
      κ η : ProbabilityTheory.Kernel α γ
      inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
      inst✝ : ProbabilityTheory.IsFiniteKernel η
      a : α
      h : Eq ((κ.singularPart η) a) 0
      ⊢ ((η.withDensity (κ.rnDeriv η)) a).AbsolutelyContinuous (η a)
    -/
    exact withDensity_absolutelyContinuous _ _
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    h : (HAdd.hAdd ((η.withDensity (κ.rnDeriv η)) a) ((κ.singularPart η) a)).Absol …
    ⊢ Eq ((κ.singularPart η) a) 0
  -/
  rw [Measure.AbsolutelyContinuous.add_left_iff] at h
  exact Measure.eq_zero_of_absolutelyContinuous_of_mutuallySingular h.2
    (mutuallySingular_singularPart _ _ _)


lemma withDensity_rnDeriv_eq_zero_iff_mutuallySingular (κ η : Kernel α γ)
    [IsFiniteKernel κ] [IsFiniteKernel η] (a : α) :
    withDensity η (rnDeriv κ η) a = 0 ↔ κ a ⟂ₘ η a := by
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    ⊢ Iff (Eq ((η.withDensity (κ.rnDeriv η)) a) 0) ((κ a).MutuallySingular (η a))
  -/
  conv_rhs => rw [← rnDeriv_add_singularPart κ η, coe_add, Pi.add_apply]
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    ⊢ Iff (Eq ((η.withDensity (κ.rnDeriv η)) a) 0) ((HAdd.hAdd ((η.withDensity (κ. …
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ ?_⟩
    /-
      case refine_1
      α : Type u_1
      γ : Type u_2
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
      κ η : ProbabilityTheory.Kernel α γ
      inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
      inst✝ : ProbabilityTheory.IsFiniteKernel η
      a : α
      h : Eq ((η.withDensity (κ.rnDeriv η)) a) 0
      ⊢ (HAdd.hAdd ((η.withDensity (κ.rnDeriv η)) a) ((κ.singularPart η) a)).Mutuall …
    -/
  · rw [h, zero_add]
    /-
      case refine_1
      α : Type u_1
      γ : Type u_2
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
      κ η : ProbabilityTheory.Kernel α γ
      inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
      inst✝ : ProbabilityTheory.IsFiniteKernel η
      a : α
      h : Eq ((η.withDensity (κ.rnDeriv η)) a) 0
      ⊢ ((κ.singularPart η) a).MutuallySingular (η a)
    -/
    exact mutuallySingular_singularPart _ _ _
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    h : (HAdd.hAdd ((η.withDensity (κ.rnDeriv η)) a) ((κ.singularPart η) a)).Mutua …
    ⊢ Eq ((η.withDensity (κ.rnDeriv η)) a) 0
  -/
  rw [Measure.MutuallySingular.add_left_iff] at h
  /-
    case refine_2
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    h : And (((η.withDensity (κ.rnDeriv η)) a).MutuallySingular (η a)) (((κ.singul …
    ⊢ Eq ((η.withDensity (κ.rnDeriv η)) a) 0
  -/
  rw [← Measure.MutuallySingular.self_iff]
  exact h.1.mono_ac Measure.AbsolutelyContinuous.rfl
    (withDensity_absolutelyContinuous (κ := η) (rnDeriv κ η) a)


lemma singularPart_eq_zero_iff_measure_eq_zero (κ η : Kernel α γ)
    [IsFiniteKernel κ] [IsFiniteKernel η] (a : α) :
    singularPart κ η a = 0 ↔ κ a (mutuallySingularSetSlice κ η a) = 0 := by
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    ⊢ Iff (Eq ((κ.singularPart η) a) 0) (Eq ((κ a) (κ.mutuallySingularSetSlice η a …
  -/
  have h_eq_add := rnDeriv_add_singularPart κ η
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    h_eq_add : Eq (HAdd.hAdd (η.withDensity (κ.rnDeriv η)) (κ.singularPart η)) κ
    ⊢ Iff (Eq ((κ.singularPart η) a) 0) (Eq ((κ a) (κ.mutuallySingularSetSlice η a …
  -/
  simp_rw [Kernel.ext_iff, Measure.ext_iff] at h_eq_add
  specialize h_eq_add a (mutuallySingularSetSlice κ η a)
    (measurableSet_mutuallySingularSetSlice κ η a)
  simp only [coe_add, Pi.add_apply, Measure.coe_add,
    withDensity_rnDeriv_mutuallySingularSetSlice κ η, zero_add] at h_eq_add
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    h_eq_add : Eq (((κ.singularPart η) a) (κ.mutuallySingularSetSlice η a)) ((κ a) …
    ⊢ Iff (Eq ((κ.singularPart η) a) 0) (Eq ((κ a) (κ.mutuallySingularSetSlice η a …
  -/
  rw [← h_eq_add]
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    h_eq_add : Eq (((κ.singularPart η) a) (κ.mutuallySingularSetSlice η a)) ((κ a) …
    ⊢ Iff (Eq ((κ.singularPart η) a) 0) (Eq (((κ.singularPart η) a) (κ.mutuallySin …
  -/
  exact singularPart_eq_zero_iff_apply_eq_zero κ η a
  /-
    🎉 no goals
  -/


lemma withDensity_rnDeriv_eq_zero_iff_measure_eq_zero (κ η : Kernel α γ)
    [IsFiniteKernel κ] [IsFiniteKernel η] (a : α) :
    withDensity η (rnDeriv κ η) a = 0 ↔ κ a (mutuallySingularSetSlice κ η a)ᶜ = 0 := by
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    ⊢ Iff (Eq ((η.withDensity (κ.rnDeriv η)) a) 0) (Eq ((κ a) (HasCompl.compl (κ.m …
  -/
  have h_eq_add := rnDeriv_add_singularPart κ η
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    h_eq_add : Eq (HAdd.hAdd (η.withDensity (κ.rnDeriv η)) (κ.singularPart η)) κ
    ⊢ Iff (Eq ((η.withDensity (κ.rnDeriv η)) a) 0) (Eq ((κ a) (HasCompl.compl (κ.m …
  -/
  simp_rw [Kernel.ext_iff, Measure.ext_iff] at h_eq_add
  specialize h_eq_add a (mutuallySingularSetSlice κ η a)ᶜ
    (measurableSet_mutuallySingularSetSlice κ η a).compl
  simp only [coe_add, Pi.add_apply, Measure.coe_add,
    singularPart_compl_mutuallySingularSetSlice κ η, add_zero] at h_eq_add
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    h_eq_add : Eq (((η.withDensity (κ.rnDeriv η)) a) (HasCompl.compl (κ.mutuallySi …
    ⊢ Iff (Eq ((η.withDensity (κ.rnDeriv η)) a) 0) (Eq ((κ a) (HasCompl.compl (κ.m …
  -/
  rw [← h_eq_add]
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    h_eq_add : Eq (((η.withDensity (κ.rnDeriv η)) a) (HasCompl.compl (κ.mutuallySi …
    ⊢ Iff (Eq ((η.withDensity (κ.rnDeriv η)) a) 0) (Eq (((η.withDensity (κ.rnDeriv …
  -/
  exact withDensity_rnDeriv_eq_zero_iff_apply_eq_zero κ η a
  /-
    🎉 no goals
  -/


/-- The set of points `a : α` such that `κ a ≪ η a` is measurable. -/
@[measurability]
lemma measurableSet_absolutelyContinuous (κ η : Kernel α γ) [IsFiniteKernel κ] [IsFiniteKernel η] :
    MeasurableSet {a | κ a ≪ η a} := by
  simp_rw [← singularPart_eq_zero_iff_absolutelyContinuous,
    singularPart_eq_zero_iff_measure_eq_zero]
  exact measurable_kernel_prod_mk_left (measurableSet_mutuallySingularSet κ η)
    (measurableSet_singleton 0)


/-- The set of points `a : α` such that `κ a ⟂ₘ η a` is measurable. -/
@[measurability]
lemma measurableSet_mutuallySingular (κ η : Kernel α γ) [IsFiniteKernel κ] [IsFiniteKernel η] :
    MeasurableSet {a | κ a ⟂ₘ η a} := by
  simp_rw [← withDensity_rnDeriv_eq_zero_iff_mutuallySingular,
    withDensity_rnDeriv_eq_zero_iff_measure_eq_zero]
  exact measurable_kernel_prod_mk_left (measurableSet_mutuallySingularSet κ η).compl
    (measurableSet_singleton 0)


@[simp]
lemma singularPart_self (κ : Kernel α γ) [IsFiniteKernel κ] : κ.singularPart κ = 0 := by
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ : ProbabilityTheory.Kernel α γ
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    ⊢ Eq (κ.singularPart κ) 0
  -/
  ext : 1; rw [zero_apply, singularPart_eq_zero_iff_absolutelyContinuous]
           /-
             🎉 no goals
           -/


omit hαγ in
lemma eq_rnDeriv_measure (h : κ = η.withDensity f + ξ)
    (hf : Measurable (Function.uncurry f)) (a : α) (hξ : ξ a ⟂ₘ η a) :
    f a =ᵐ[η a] ∂(κ a)/∂(η a) := by
  have : κ a = ξ a + (η a).withDensity (f a) := by
    rw [h, coe_add, Pi.add_apply, η.withDensity_apply hf, add_comm]
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    κ η ξ : ProbabilityTheory.Kernel α γ
    f : α → γ → ENNReal
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    h : Eq κ (HAdd.hAdd (η.withDensity f) ξ)
    hf : Measurable (Function.uncurry f)
    a : α
    hξ : (ξ a).MutuallySingular (η a)
    this : Eq (κ a) (HAdd.hAdd (ξ a) ((η a).withDensity (f a)))
    ⊢ (MeasureTheory.ae (η a)).EventuallyEq (f a) ((κ a).rnDeriv (η a))
  -/
  exact (κ a).eq_rnDeriv₀ (hf.comp measurable_prod_mk_left).aemeasurable hξ this
  /-
    🎉 no goals
  -/


omit hαγ in
lemma eq_singularPart_measure (h : κ = η.withDensity f + ξ)
    (hf : Measurable (Function.uncurry f)) (a : α) (hξ : ξ a ⟂ₘ η a) :
    ξ a = (κ a).singularPart (η a) := by
  have : κ a = ξ a + (η a).withDensity (f a) := by
    rw [h, coe_add, Pi.add_apply, η.withDensity_apply hf, add_comm]
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    κ η ξ : ProbabilityTheory.Kernel α γ
    f : α → γ → ENNReal
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    h : Eq κ (HAdd.hAdd (η.withDensity f) ξ)
    hf : Measurable (Function.uncurry f)
    a : α
    hξ : (ξ a).MutuallySingular (η a)
    this : Eq (κ a) (HAdd.hAdd (ξ a) ((η a).withDensity (f a)))
    ⊢ Eq (ξ a) ((κ a).singularPart (η a))
  -/
  exact (κ a).eq_singularPart (hf.comp measurable_prod_mk_left) hξ this
  /-
    🎉 no goals
  -/


lemma rnDeriv_eq_rnDeriv_measure : rnDeriv κ η a =ᵐ[η a] ∂(κ a)/∂(η a) :=
  eq_rnDeriv_measure (rnDeriv_add_singularPart κ η).symm (measurable_rnDeriv κ η) a
    (mutuallySingular_singularPart κ η a)


lemma singularPart_eq_singularPart_measure : singularPart κ η a = (κ a).singularPart (η a) :=
  eq_singularPart_measure (rnDeriv_add_singularPart κ η).symm (measurable_rnDeriv κ η) a
    (mutuallySingular_singularPart κ η a)


lemma eq_rnDeriv (h : κ = η.withDensity f + ξ)
    (hf : Measurable (Function.uncurry f)) (a : α) (hξ : ξ a ⟂ₘ η a) :
    f a =ᵐ[η a] rnDeriv κ η a :=
  (eq_rnDeriv_measure h hf a hξ).trans rnDeriv_eq_rnDeriv_measure.symm


lemma eq_singularPart (h : κ = η.withDensity f + ξ)
    (hf : Measurable (Function.uncurry f)) (a : α) (hξ : ξ a ⟂ₘ η a) :
    ξ a = singularPart κ η a :=
  (eq_singularPart_measure h hf a hξ).trans singularPart_eq_singularPart_measure.symm


instance [hκ : IsFiniteKernel κ] [IsFiniteKernel η] :
    IsFiniteKernel (withDensity η (rnDeriv κ η)) := by
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    κ η : ProbabilityTheory.Kernel α γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    hκ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    ⊢ ProbabilityTheory.IsFiniteKernel (η.withDensity (κ.rnDeriv η))
  -/
  refine ⟨hκ.bound, hκ.bound_lt_top, fun a ↦ ?_⟩
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    κ η : ProbabilityTheory.Kernel α γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    hκ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    ⊢ LE.le (((η.withDensity (κ.rnDeriv η)) a) Set.univ) (ProbabilityTheory.IsFini …
  -/
  rw [Kernel.withDensity_apply', setLIntegral_univ]
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    κ η : ProbabilityTheory.Kernel α γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    hκ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    ⊢ LE.le (MeasureTheory.lintegral (η a) fun x => κ.rnDeriv η a x) (ProbabilityT …
  -/
  swap; · exact measurable_rnDeriv κ η
          /-
            🎉 no goals
          -/
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    κ η : ProbabilityTheory.Kernel α γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    hκ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    ⊢ LE.le (MeasureTheory.lintegral (η a) fun x => κ.rnDeriv η a x) (ProbabilityT …
  -/
  rw [lintegral_congr_ae rnDeriv_eq_rnDeriv_measure]
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    κ η : ProbabilityTheory.Kernel α γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    hκ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    ⊢ LE.le (MeasureTheory.lintegral (η a) fun a_1 => (κ a).rnDeriv (η a) a_1) (Pr …
  -/
  exact Measure.lintegral_rnDeriv_le.trans (measure_le_bound _ _ _)
  /-
    🎉 no goals
  -/


instance [hκ : IsFiniteKernel κ] [IsFiniteKernel η] : IsFiniteKernel (singularPart κ η) := by
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    κ η : ProbabilityTheory.Kernel α γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    hκ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    ⊢ ProbabilityTheory.IsFiniteKernel (κ.singularPart η)
  -/
  refine ⟨hκ.bound, hκ.bound_lt_top, fun a ↦ ?_⟩
  have h : withDensity η (rnDeriv κ η) a univ + singularPart κ η a univ = κ a univ := by
    conv_rhs => rw [← rnDeriv_add_singularPart κ η]
    simp
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    κ η : ProbabilityTheory.Kernel α γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    hκ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    h : Eq (HAdd.hAdd (((η.withDensity (κ.rnDeriv η)) a) Set.univ) (((κ.singularPa …
    ⊢ LE.le (((κ.singularPart η) a) Set.univ) (ProbabilityTheory.IsFiniteKernel.bo …
  -/
  exact (self_le_add_left _ _).trans (h.le.trans (measure_le_bound _ _ _))
  /-
    🎉 no goals
  -/


/-- For two kernels `κ, η`, the singular part of `κ a` with respect to `η a` is a measurable
function of `a`. -/
lemma measurable_singularPart (κ η : Kernel α γ) [IsFiniteKernel κ] [IsFiniteKernel η] :
    Measurable (fun a ↦ (κ a).singularPart (η a)) := by
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    ⊢ Measurable fun a => (κ a).singularPart (η a)
  -/
  refine Measure.measurable_of_measurable_coe _ (fun s hs ↦ ?_)
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    s : Set γ
    hs : MeasurableSet s
    ⊢ Measurable fun b => ((κ b).singularPart (η b)) s
  -/
  simp_rw [← κ.singularPart_eq_singularPart_measure, κ.singularPart_def η]
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    s : Set γ
    hs : MeasurableSet s
    ⊢ Measurable fun b => (((HAdd.hAdd κ η).withDensity fun a x => HSub.hSub (↑(κ. …
  -/
  exact Kernel.measurable_coe _ hs
  /-
    🎉 no goals
  -/


lemma rnDeriv_self (κ : Kernel α γ) [IsFiniteKernel κ] (a : α) : rnDeriv κ κ a =ᵐ[κ a] 1 :=
  (κ.rnDeriv_eq_rnDeriv_measure).trans (κ a).rnDeriv_self


lemma rnDeriv_singularPart (κ ν : Kernel α γ) [IsFiniteKernel κ] [IsFiniteKernel ν] (a : α) :
    rnDeriv (singularPart κ ν) ν a =ᵐ[ν a] 0 := by
  filter_upwards [(singularPart κ ν).rnDeriv_eq_rnDeriv_measure,
    (Measure.rnDeriv_eq_zero _ _).mpr (mutuallySingular_singularPart κ ν a)] with x h1 h2
  /-
    case h
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ ν : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    a : α
    x : γ
    h1 : Eq ((κ.singularPart ν).rnDeriv ν a x) (((κ.singularPart ν) a).rnDeriv (ν  …
    h2 : Eq (((κ.singularPart ν) a).rnDeriv (ν a) x) (0 x)
    ⊢ Eq ((κ.singularPart ν).rnDeriv ν a x) (0 x)
  -/
  rw [h1, h2]
  /-
    🎉 no goals
  -/


lemma rnDeriv_lt_top (κ η : Kernel α γ) [IsFiniteKernel κ] [IsFiniteKernel η] {a : α} :
    ∀ᵐ x ∂(η a), rnDeriv κ η a x < ∞ := by
  filter_upwards [κ.rnDeriv_eq_rnDeriv_measure, (κ a).rnDeriv_ne_top _]
    with x heq htop using heq ▸ htop.lt_top


lemma rnDeriv_ne_top (κ η : Kernel α γ) [IsFiniteKernel κ] [IsFiniteKernel η] {a : α} :
    ∀ᵐ x ∂(η a), rnDeriv κ η a x ≠ ∞ := by
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    ⊢ Filter.Eventually (fun x => Ne (κ.rnDeriv η a x) Top.top) (MeasureTheory.ae  …
  -/
  filter_upwards [κ.rnDeriv_lt_top η] with a h using h.ne
  /-
    🎉 no goals
  -/


lemma rnDeriv_pos [IsFiniteKernel κ] [IsFiniteKernel η] {a : α} (ha : κ a ≪ η a) :
    ∀ᵐ x ∂(κ a), 0 < rnDeriv κ η a x := by
  filter_upwards [ha.ae_le κ.rnDeriv_eq_rnDeriv_measure, Measure.rnDeriv_pos ha]
    with x heq hpos using heq ▸ hpos


lemma rnDeriv_toReal_pos [IsFiniteKernel κ] [IsFiniteKernel η] {a : α} (h : κ a ≪ η a) :
    ∀ᵐ x ∂(κ a), 0 < (rnDeriv κ η a x).toReal := by
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    κ η : ProbabilityTheory.Kernel α γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    h : (κ a).AbsolutelyContinuous (η a)
    ⊢ Filter.Eventually (fun x => LT.lt 0 (κ.rnDeriv η a x).toReal) (MeasureTheory …
  -/
  filter_upwards [rnDeriv_pos h, h.ae_le (rnDeriv_ne_top κ _)] with x h0 htop
  /-
    case h
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    κ η : ProbabilityTheory.Kernel α γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    h : (κ a).AbsolutelyContinuous (η a)
    x : γ
    h0 : LT.lt 0 (κ.rnDeriv η a x)
    htop : Ne (κ.rnDeriv η a x) Top.top
    ⊢ LT.lt 0 (κ.rnDeriv η a x).toReal
  -/
  simp_all only [pos_iff_ne_zero, ne_eq, ENNReal.toReal_pos, not_false_eq_true, and_self]
  /-
    🎉 no goals
  -/


lemma rnDeriv_add (κ ν η : Kernel α γ) [IsFiniteKernel κ] [IsFiniteKernel ν] [IsFiniteKernel η]
    (a : α) :
    rnDeriv (κ + ν) η a =ᵐ[η a] rnDeriv κ η a + rnDeriv ν η a := by
  filter_upwards [(κ + ν).rnDeriv_eq_rnDeriv_measure, κ.rnDeriv_eq_rnDeriv_measure,
    ν.rnDeriv_eq_rnDeriv_measure, (κ a).rnDeriv_add (ν a) (η a)] with x h1 h2 h3 h4
  /-
    case h
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ ν η : ProbabilityTheory.Kernel α γ
    inst✝² : ProbabilityTheory.IsFiniteKernel κ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel ν
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    x : γ
    h1 : Eq ((HAdd.hAdd κ ν).rnDeriv η a x) (((HAdd.hAdd κ ν) a).rnDeriv (η a) x)
    h2 : Eq (κ.rnDeriv η a x) ((κ a).rnDeriv (η a) x)
    h3 : Eq (ν.rnDeriv η a x) ((ν a).rnDeriv (η a) x)
    h4 : Eq ((HAdd.hAdd (κ a) (ν a)).rnDeriv (η a) x) (HAdd.hAdd ((κ a).rnDeriv (η …
    ⊢ Eq ((HAdd.hAdd κ ν).rnDeriv η a x) (HAdd.hAdd (κ.rnDeriv η a) (ν.rnDeriv η a …
  -/
  rw [h1, Pi.add_apply, h2, h3, coe_add, Pi.add_apply, h4, Pi.add_apply]
  /-
    🎉 no goals
  -/


lemma withDensity_rnDeriv_le (κ η : Kernel α γ) [IsFiniteKernel κ] [IsFiniteKernel η] (a : α) :
    η.withDensity (κ.rnDeriv η) a ≤ κ a := by
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    ⊢ LE.le ((η.withDensity (κ.rnDeriv η)) a) (κ a)
  -/
  refine Measure.le_intro (fun s hs _ ↦ ?_)
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    s : Set γ
    hs : MeasurableSet s
    x✝ : s.Nonempty
    ⊢ LE.le (((η.withDensity (κ.rnDeriv η)) a) s) ((κ a) s)
  -/
  rw [Kernel.withDensity_apply']
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    s : Set γ
    hs : MeasurableSet s
    x✝ : s.Nonempty
    ⊢ LE.le (MeasureTheory.lintegral ((η a).restrict s) fun b => κ.rnDeriv η a b)  …
  -/
  swap; · exact κ.measurable_rnDeriv _
          /-
            🎉 no goals
          -/
  rw [setLIntegral_congr_fun hs ((κ.rnDeriv_eq_rnDeriv_measure).mono (fun x hx _ ↦ hx)),
    ← withDensity_apply _ hs]
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    κ η : ProbabilityTheory.Kernel α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    s : Set γ
    hs : MeasurableSet s
    x✝ : s.Nonempty
    ⊢ LE.le (((η a).withDensity ((κ a).rnDeriv (η a))) s) ((κ a) s)
  -/
  exact (κ a).withDensity_rnDeriv_le _ _
  /-
    🎉 no goals
  -/


lemma withDensity_rnDeriv_eq [IsFiniteKernel κ] [IsFiniteKernel η] {a : α} (h : κ a ≪ η a) :
    η.withDensity (κ.rnDeriv η) a = κ a := by
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    κ η : ProbabilityTheory.Kernel α γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    h : (κ a).AbsolutelyContinuous (η a)
    ⊢ Eq ((η.withDensity (κ.rnDeriv η)) a) (κ a)
  -/
  rw [Kernel.withDensity_apply]
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    κ η : ProbabilityTheory.Kernel α γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    h : (κ a).AbsolutelyContinuous (η a)
    ⊢ Eq ((η a).withDensity (κ.rnDeriv η a)) (κ a)
  -/
  swap; · exact κ.measurable_rnDeriv _
          /-
            🎉 no goals
          -/
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    κ η : ProbabilityTheory.Kernel α γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    h : (κ a).AbsolutelyContinuous (η a)
    ⊢ Eq ((η a).withDensity (κ.rnDeriv η a)) (κ a)
  -/
  have h_ae := κ.rnDeriv_eq_rnDeriv_measure (η := η) (a := a)
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    κ η : ProbabilityTheory.Kernel α γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    h : (κ a).AbsolutelyContinuous (η a)
    h_ae : (MeasureTheory.ae (η a)).EventuallyEq (κ.rnDeriv η a) ((κ a).rnDeriv (η …
    ⊢ Eq ((η a).withDensity (κ.rnDeriv η a)) (κ a)
  -/
  rw [MeasureTheory.withDensity_congr_ae h_ae, (κ a).withDensity_rnDeriv_eq _ h]
  /-
    🎉 no goals
  -/


lemma rnDeriv_withDensity [IsFiniteKernel κ] {f : α → γ → ℝ≥0∞} [IsFiniteKernel (withDensity κ f)]
    (hf : Measurable (Function.uncurry f)) (a : α) :
    (κ.withDensity f).rnDeriv κ a =ᵐ[κ a] f a := by
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    f : α → γ → ENNReal
    inst✝ : ProbabilityTheory.IsFiniteKernel (κ.withDensity f)
    hf : Measurable (Function.uncurry f)
    a : α
    ⊢ (MeasureTheory.ae (κ a)).EventuallyEq ((κ.withDensity f).rnDeriv κ a) (f a)
  -/
  have h_ae := (κ.withDensity f).rnDeriv_eq_rnDeriv_measure (η := κ) (a := a)
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    f : α → γ → ENNReal
    inst✝ : ProbabilityTheory.IsFiniteKernel (κ.withDensity f)
    hf : Measurable (Function.uncurry f)
    a : α
    h_ae : (MeasureTheory.ae (κ a)).EventuallyEq ((κ.withDensity f).rnDeriv κ a) ( …
    ⊢ (MeasureTheory.ae (κ a)).EventuallyEq ((κ.withDensity f).rnDeriv κ a) (f a)
  -/
  have hf' : ∀ a, Measurable (f a) := fun _ ↦ hf.of_uncurry_left
  /-
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    f : α → γ → ENNReal
    inst✝ : ProbabilityTheory.IsFiniteKernel (κ.withDensity f)
    hf : Measurable (Function.uncurry f)
    a : α
    h_ae : (MeasureTheory.ae (κ a)).EventuallyEq ((κ.withDensity f).rnDeriv κ a) ( …
    hf' : ∀ (a : α), Measurable (f a)
    ⊢ (MeasureTheory.ae (κ a)).EventuallyEq ((κ.withDensity f).rnDeriv κ a) (f a)
  -/
  filter_upwards [h_ae, (κ a).rnDeriv_withDensity (hf' a)] with x hx1 hx2
  /-
    case h
    α : Type u_1
    γ : Type u_2
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α γ
    hαγ : MeasurableSpace.CountableOrCountablyGenerated α γ
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    f : α → γ → ENNReal
    inst✝ : ProbabilityTheory.IsFiniteKernel (κ.withDensity f)
    hf : Measurable (Function.uncurry f)
    a : α
    h_ae : (MeasureTheory.ae (κ a)).EventuallyEq ((κ.withDensity f).rnDeriv κ a) ( …
    hf' : ∀ (a : α), Measurable (f a)
    x : γ
    hx1 : Eq ((κ.withDensity f).rnDeriv κ a x) (((κ.withDensity f) a).rnDeriv (κ a …
    hx2 : Eq (((κ a).withDensity (f a)).rnDeriv (κ a) x) (f a x)
    ⊢ Eq ((κ.withDensity f).rnDeriv κ a x) (f a x)
  -/
  rw [hx1, κ.withDensity_apply hf, hx2]
  /-
    🎉 no goals
  -/


