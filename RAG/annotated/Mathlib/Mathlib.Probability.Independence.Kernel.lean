/-- A family of sets of sets `π : ι → Set (Set Ω)` is independent with respect to a kernel `κ` and
a measure `μ` if for any finite set of indices `s = {i_1, ..., i_n}`, for any sets
`f i_1 ∈ π i_1, ..., f i_n ∈ π i_n`, then `∀ᵐ a ∂μ, κ a (⋂ i in s, f i) = ∏ i ∈ s, κ a (f i)`.
It will be used for families of pi_systems. -/
def iIndepSets {_mΩ : MeasurableSpace Ω}
    (π : ι → Set (Set Ω)) (κ : Kernel α Ω) (μ : Measure α := by volume_tac) : Prop :=
  ∀ (s : Finset ι) {f : ι → Set Ω} (_H : ∀ i, i ∈ s → f i ∈ π i),
  ∀ᵐ a ∂μ, κ a (⋂ i ∈ s, f i) = ∏ i ∈ s, κ a (f i)


/-- Two sets of sets `s₁, s₂` are independent with respect to a kernel `κ` and a measure `μ` if for
any sets `t₁ ∈ s₁, t₂ ∈ s₂`, then `∀ᵐ a ∂μ, κ a (t₁ ∩ t₂) = κ a (t₁) * κ a (t₂)` -/
def IndepSets {_mΩ : MeasurableSpace Ω}
    (s1 s2 : Set (Set Ω)) (κ : Kernel α Ω) (μ : Measure α := by volume_tac) : Prop :=
  ∀ t1 t2 : Set Ω, t1 ∈ s1 → t2 ∈ s2 → (∀ᵐ a ∂μ, κ a (t1 ∩ t2) = κ a t1 * κ a t2)


/-- A family of measurable space structures (i.e. of σ-algebras) is independent with respect to a
kernel `κ` and a measure `μ` if the family of sets of measurable sets they define is independent. -/
def iIndep (m : ι → MeasurableSpace Ω) {_mΩ : MeasurableSpace Ω} (κ : Kernel α Ω)
    (μ : Measure α := by volume_tac) : Prop :=
  iIndepSets (fun x ↦ {s | MeasurableSet[m x] s}) κ μ


/-- Two measurable space structures (or σ-algebras) `m₁, m₂` are independent with respect to a
kernel `κ` and a measure `μ` if for any sets `t₁ ∈ m₁, t₂ ∈ m₂`,
`∀ᵐ a ∂μ, κ a (t₁ ∩ t₂) = κ a (t₁) * κ a (t₂)` -/
def Indep (m₁ m₂ : MeasurableSpace Ω) {_mΩ : MeasurableSpace Ω} (κ : Kernel α Ω)
    (μ : Measure α := by volume_tac) : Prop :=
  IndepSets {s | MeasurableSet[m₁] s} {s | MeasurableSet[m₂] s} κ μ


/-- A family of sets is independent if the family of measurable space structures they generate is
independent. For a set `s`, the generated measurable space has measurable sets `∅, s, sᶜ, univ`. -/
def iIndepSet {_mΩ : MeasurableSpace Ω} (s : ι → Set Ω) (κ : Kernel α Ω)
    (μ : Measure α := by volume_tac) : Prop :=
  iIndep (fun i ↦ generateFrom {s i}) κ μ


/-- Two sets are independent if the two measurable space structures they generate are independent.
For a set `s`, the generated measurable space structure has measurable sets `∅, s, sᶜ, univ`. -/
def IndepSet {_mΩ : MeasurableSpace Ω} (s t : Set Ω) (κ : Kernel α Ω)
    (μ : Measure α := by volume_tac) : Prop :=
  Indep (generateFrom {s}) (generateFrom {t}) κ μ


/-- A family of functions defined on the same space `Ω` and taking values in possibly different
spaces, each with a measurable space structure, is independent if the family of measurable space
structures they generate on `Ω` is independent. For a function `g` with codomain having measurable
space structure `m`, the generated measurable space structure is `MeasurableSpace.comap g m`. -/
def iIndepFun {_mΩ : MeasurableSpace Ω} {β : ι → Type*} (m : ∀ x : ι, MeasurableSpace (β x))
    (f : ∀ x : ι, Ω → β x) (κ : Kernel α Ω)
    (μ : Measure α := by volume_tac) : Prop :=
  iIndep (fun x ↦ MeasurableSpace.comap (f x) (m x)) κ μ


/-- Two functions are independent if the two measurable space structures they generate are
independent. For a function `f` with codomain having measurable space structure `m`, the generated
measurable space structure is `MeasurableSpace.comap f m`. -/
def IndepFun {β γ} {_mΩ : MeasurableSpace Ω} [mβ : MeasurableSpace β] [mγ : MeasurableSpace γ]
    (f : Ω → β) (g : Ω → γ) (κ : Kernel α Ω)
    (μ : Measure α := by volume_tac) : Prop :=
  Indep (MeasurableSpace.comap f mβ) (MeasurableSpace.comap g mγ) κ μ


                                                             /-
                                                               α : Type u_1
                                                               Ω : Type u_2
                                                               ι : Type u_3
                                                               _mα : MeasurableSpace α
                                                               _mΩ : MeasurableSpace Ω
                                                               κ : ProbabilityTheory.Kernel α Ω
                                                               π : ι → Set (Set Ω)
                                                               ⊢ ProbabilityTheory.Kernel.iIndepSets π κ 0
                                                             -/
@[simp] lemma iIndepSets_zero_right : iIndepSets π κ 0 := by simp [iIndepSets]
                                                             /-
                                                               🎉 no goals
                                                             -/


                                                               /-
                                                                 α : Type u_1
                                                                 Ω : Type u_2
                                                                 _mα : MeasurableSpace α
                                                                 _mΩ : MeasurableSpace Ω
                                                                 κ : ProbabilityTheory.Kernel α Ω
                                                                 s1 s2 : Set (Set Ω)
                                                                 ⊢ ProbabilityTheory.Kernel.IndepSets s1 s2 κ 0
                                                               -/
@[simp] lemma indepSets_zero_right : IndepSets s1 s2 κ 0 := by simp [IndepSets]
                                                               /-
                                                                 🎉 no goals
                                                               -/


                                                                             /-
                                                                               α : Type u_1
                                                                               Ω : Type u_2
                                                                               _mα : MeasurableSpace α
                                                                               _mΩ : MeasurableSpace Ω
                                                                               μ : MeasureTheory.Measure α
                                                                               s1 s2 : Set (Set Ω)
                                                                               ⊢ ProbabilityTheory.Kernel.IndepSets s1 s2 0 μ
                                                                             -/
@[simp] lemma indepSets_zero_left : IndepSets s1 s2 (0 : Kernel α Ω) μ := by simp [IndepSets]
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


                                                     /-
                                                       α : Type u_1
                                                       Ω : Type u_2
                                                       ι : Type u_3
                                                       _mα : MeasurableSpace α
                                                       m : ι → MeasurableSpace Ω
                                                       _mΩ : MeasurableSpace Ω
                                                       κ : ProbabilityTheory.Kernel α Ω
                                                       ⊢ ProbabilityTheory.Kernel.iIndep m κ 0
                                                     -/
@[simp] lemma iIndep_zero_right : iIndep m κ 0 := by simp [iIndep]
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp] lemma indep_zero_right {m₁ m₂ : MeasurableSpace Ω} {_mΩ : MeasurableSpace Ω}
                                             /-
                                               α : Type u_1
                                               Ω : Type u_2
                                               _mα : MeasurableSpace α
                                               m₁ m₂ _mΩ : MeasurableSpace Ω
                                               κ : ProbabilityTheory.Kernel α Ω
                                               ⊢ ProbabilityTheory.Kernel.Indep m₁ m₂ κ 0
                                             -/
    {κ : Kernel α Ω} : Indep m₁ m₂ κ 0 := by simp [Indep]
                                             /-
                                               🎉 no goals
                                             -/


@[simp] lemma indep_zero_left {m₁ m₂ : MeasurableSpace Ω} {_mΩ : MeasurableSpace Ω} :
                                          /-
                                            α : Type u_1
                                            Ω : Type u_2
                                            _mα : MeasurableSpace α
                                            μ : MeasureTheory.Measure α
                                            m₁ m₂ _mΩ : MeasurableSpace Ω
                                            ⊢ ProbabilityTheory.Kernel.Indep m₁ m₂ 0 μ
                                          -/
    Indep m₁ m₂ (0 : Kernel α Ω) μ  := by simp [Indep]
                                          /-
                                            🎉 no goals
                                          -/


                                                           /-
                                                             α : Type u_1
                                                             Ω : Type u_2
                                                             ι : Type u_3
                                                             _mα : MeasurableSpace α
                                                             _mΩ : MeasurableSpace Ω
                                                             κ : ProbabilityTheory.Kernel α Ω
                                                             s : ι → Set Ω
                                                             ⊢ ProbabilityTheory.Kernel.iIndepSet s κ 0
                                                           -/
@[simp] lemma iIndepSet_zero_right : iIndepSet s κ 0 := by simp [iIndepSet]
                                                           /-
                                                             🎉 no goals
                                                           -/


                                                                         /-
                                                                           α : Type u_1
                                                                           Ω : Type u_2
                                                                           _mα : MeasurableSpace α
                                                                           _mΩ : MeasurableSpace Ω
                                                                           κ : ProbabilityTheory.Kernel α Ω
                                                                           s t : Set Ω
                                                                           ⊢ ProbabilityTheory.Kernel.IndepSet s t κ 0
                                                                         -/
@[simp] lemma indepSet_zero_right {s t : Set Ω} : IndepSet s t κ 0 := by simp [IndepSet]
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


@[simp] lemma indepSet_zero_left {s t : Set Ω} : IndepSet s t (0 : Kernel α Ω) μ := by
  /-
    α : Type u_1
    Ω : Type u_2
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure α
    s t : Set Ω
    ⊢ ProbabilityTheory.Kernel.IndepSet s t 0 μ
  -/
  simp [IndepSet]
  /-
    🎉 no goals
  -/


@[simp] lemma iIndepFun_zero_right {β : ι → Type*} {m : ∀ x : ι, MeasurableSpace (β x)}
                                                     /-
                                                       α : Type u_1
                                                       Ω : Type u_2
                                                       ι : Type u_3
                                                       _mα : MeasurableSpace α
                                                       _mΩ : MeasurableSpace Ω
                                                       κ : ProbabilityTheory.Kernel α Ω
                                                       β : ι → Type u_5
                                                       m : (x : ι) → MeasurableSpace (β x)
                                                       f : (x : ι) → Ω → β x
                                                       ⊢ ProbabilityTheory.Kernel.iIndepFun m f κ 0
                                                     -/
    {f : ∀ x : ι, Ω → β x} : iIndepFun m f κ 0 := by simp [iIndepFun]
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp] lemma indepFun_zero_right {β γ} [MeasurableSpace β] [MeasurableSpace γ]
                                                     /-
                                                       α : Type u_1
                                                       Ω : Type u_2
                                                       _mα : MeasurableSpace α
                                                       _mΩ : MeasurableSpace Ω
                                                       κ : ProbabilityTheory.Kernel α Ω
                                                       β : Type u_5
                                                       γ : Type u_6
                                                       inst✝¹ : MeasurableSpace β
                                                       inst✝ : MeasurableSpace γ
                                                       f : Ω → β
                                                       g : Ω → γ
                                                       ⊢ ProbabilityTheory.Kernel.IndepFun f g κ 0
                                                     -/
    {f : Ω → β} {g : Ω → γ} : IndepFun f g κ 0 := by simp [IndepFun]
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp] lemma indepFun_zero_left {β γ} [MeasurableSpace β] [MeasurableSpace γ]
                                                                    /-
                                                                      α : Type u_1
                                                                      Ω : Type u_2
                                                                      _mα : MeasurableSpace α
                                                                      _mΩ : MeasurableSpace Ω
                                                                      μ : MeasureTheory.Measure α
                                                                      β : Type u_5
                                                                      γ : Type u_6
                                                                      inst✝¹ : MeasurableSpace β
                                                                      inst✝ : MeasurableSpace γ
                                                                      f : Ω → β
                                                                      g : Ω → γ
                                                                      ⊢ ProbabilityTheory.Kernel.IndepFun f g 0 μ
                                                                    -/
    {f : Ω → β} {g : Ω → γ} : IndepFun f g (0 : Kernel α Ω) μ := by simp [IndepFun]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


lemma iIndepSets_congr (h : κ =ᵐ[μ] η) : iIndepSets π κ μ ↔ iIndepSets π η μ := by
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ η : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    π : ι → Set (Set Ω)
    h : (MeasureTheory.ae μ).EventuallyEq ⇑κ ⇑η
    ⊢ Iff (ProbabilityTheory.Kernel.iIndepSets π κ μ) (ProbabilityTheory.Kernel.iI …
  -/
  peel 3
  /-
    case h.h.h
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ η : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    π : ι → Set (Set Ω)
    h : (MeasureTheory.ae μ).EventuallyEq ⇑κ ⇑η
    a✝² : Finset ι
    a✝¹ : ι → Set Ω
    a✝ : ∀ (i : ι), Membership.mem a✝² i → Membership.mem (π i) (a✝¹ i)
    ⊢ Iff (Filter.Eventually (fun a => Eq ((κ a) (Set.iInter fun i => Set.iInter f …
  -/
  refine ⟨fun h' ↦ ?_, fun h' ↦ ?_⟩ <;>
    /-
      case h.h.h.refine_1
      α : Type u_1
      Ω : Type u_2
      ι : Type u_3
      _mα : MeasurableSpace α
      _mΩ : MeasurableSpace Ω
      κ η : ProbabilityTheory.Kernel α Ω
      μ : MeasureTheory.Measure α
      π : ι → Set (Set Ω)
      h : (MeasureTheory.ae μ).EventuallyEq ⇑κ ⇑η
      a✝² : Finset ι
      a✝¹ : ι → Set Ω
      a✝ : ∀ (i : ι), Membership.mem a✝² i → Membership.mem (π i) (a✝¹ i)
      h' : Filter.Eventually (fun a => Eq ((κ a) (Set.iInter fun i => Set.iInter fun …
      ⊢ Filter.Eventually (fun a => Eq ((η a) (Set.iInter fun i => Set.iInter fun h  …
    -/
    /-
      case h
      α : Type u_1
      Ω : Type u_2
      ι : Type u_3
      _mα : MeasurableSpace α
      _mΩ : MeasurableSpace Ω
      κ η : ProbabilityTheory.Kernel α Ω
      μ : MeasureTheory.Measure α
      π : ι → Set (Set Ω)
      h : (MeasureTheory.ae μ).EventuallyEq ⇑κ ⇑η
      a✝² : Finset ι
      a✝¹ : ι → Set Ω
      a✝ : ∀ (i : ι), Membership.mem a✝² i → Membership.mem (π i) (a✝¹ i)
      h' : Filter.Eventually (fun a => Eq ((κ a) (Set.iInter fun i => Set.iInter fun …
      a : α
      ha : Eq (κ a) (η a)
      h'a : Eq ((κ a) (Set.iInter fun i => Set.iInter fun h => a✝¹ i)) (a✝².prod fun …
      ⊢ Eq ((η a) (Set.iInter fun i => Set.iInter fun h => a✝¹ i)) (a✝².prod fun i = …
    -/
    /-
      🎉 no goals
    -/
    /-
      case h
      α : Type u_1
      Ω : Type u_2
      ι : Type u_3
      _mα : MeasurableSpace α
      _mΩ : MeasurableSpace Ω
      κ η : ProbabilityTheory.Kernel α Ω
      μ : MeasureTheory.Measure α
      π : ι → Set (Set Ω)
      h : (MeasureTheory.ae μ).EventuallyEq ⇑κ ⇑η
      a✝² : Finset ι
      a✝¹ : ι → Set Ω
      a✝ : ∀ (i : ι), Membership.mem a✝² i → Membership.mem (π i) (a✝¹ i)
      h' : Filter.Eventually (fun a => Eq ((η a) (Set.iInter fun i => Set.iInter fun …
      a : α
      ha : Eq (κ a) (η a)
      h'a : Eq ((η a) (Set.iInter fun i => Set.iInter fun h => a✝¹ i)) (a✝².prod fun …
      ⊢ Eq ((κ a) (Set.iInter fun i => Set.iInter fun h => a✝¹ i)) (a✝².prod fun i = …
    -/
    simpa [ha] using h'a
    /-
      🎉 no goals
    -/


alias ⟨iIndepSets.congr, _⟩ := iIndepSets_congr


lemma indepSets_congr (h : κ =ᵐ[μ] η) : IndepSets s1 s2 κ μ ↔ IndepSets s1 s2 η μ := by
  /-
    α : Type u_1
    Ω : Type u_2
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ η : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    s1 s2 : Set (Set Ω)
    h : (MeasureTheory.ae μ).EventuallyEq ⇑κ ⇑η
    ⊢ Iff (ProbabilityTheory.Kernel.IndepSets s1 s2 κ μ) (ProbabilityTheory.Kernel …
  -/
  peel 4
  /-
    case h.h.h.h
    α : Type u_1
    Ω : Type u_2
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ η : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    s1 s2 : Set (Set Ω)
    h : (MeasureTheory.ae μ).EventuallyEq ⇑κ ⇑η
    a✝³ a✝² : Set Ω
    a✝¹ : Membership.mem s1 a✝³
    a✝ : Membership.mem s2 a✝²
    ⊢ Iff (Filter.Eventually (fun a => Eq ((κ a) (Inter.inter a✝³ a✝²)) (HMul.hMul …
  -/
  refine ⟨fun h' ↦ ?_, fun h' ↦ ?_⟩ <;>
    /-
      case h.h.h.h.refine_1
      α : Type u_1
      Ω : Type u_2
      _mα : MeasurableSpace α
      _mΩ : MeasurableSpace Ω
      κ η : ProbabilityTheory.Kernel α Ω
      μ : MeasureTheory.Measure α
      s1 s2 : Set (Set Ω)
      h : (MeasureTheory.ae μ).EventuallyEq ⇑κ ⇑η
      a✝³ a✝² : Set Ω
      a✝¹ : Membership.mem s1 a✝³
      a✝ : Membership.mem s2 a✝²
      h' : Filter.Eventually (fun a => Eq ((κ a) (Inter.inter a✝³ a✝²)) (HMul.hMul ( …
      ⊢ Filter.Eventually (fun a => Eq ((η a) (Inter.inter a✝³ a✝²)) (HMul.hMul ((η  …
    -/
    /-
      case h
      α : Type u_1
      Ω : Type u_2
      _mα : MeasurableSpace α
      _mΩ : MeasurableSpace Ω
      κ η : ProbabilityTheory.Kernel α Ω
      μ : MeasureTheory.Measure α
      s1 s2 : Set (Set Ω)
      h : (MeasureTheory.ae μ).EventuallyEq ⇑κ ⇑η
      a✝³ a✝² : Set Ω
      a✝¹ : Membership.mem s1 a✝³
      a✝ : Membership.mem s2 a✝²
      h' : Filter.Eventually (fun a => Eq ((κ a) (Inter.inter a✝³ a✝²)) (HMul.hMul ( …
      a : α
      ha : Eq (κ a) (η a)
      h'a : Eq ((κ a) (Inter.inter a✝³ a✝²)) (HMul.hMul ((κ a) a✝³) ((κ a) a✝²))
      ⊢ Eq ((η a) (Inter.inter a✝³ a✝²)) (HMul.hMul ((η a) a✝³) ((η a) a✝²))
    -/
    /-
      🎉 no goals
    -/
    /-
      case h
      α : Type u_1
      Ω : Type u_2
      _mα : MeasurableSpace α
      _mΩ : MeasurableSpace Ω
      κ η : ProbabilityTheory.Kernel α Ω
      μ : MeasureTheory.Measure α
      s1 s2 : Set (Set Ω)
      h : (MeasureTheory.ae μ).EventuallyEq ⇑κ ⇑η
      a✝³ a✝² : Set Ω
      a✝¹ : Membership.mem s1 a✝³
      a✝ : Membership.mem s2 a✝²
      h' : Filter.Eventually (fun a => Eq ((η a) (Inter.inter a✝³ a✝²)) (HMul.hMul ( …
      a : α
      ha : Eq (κ a) (η a)
      h'a : Eq ((η a) (Inter.inter a✝³ a✝²)) (HMul.hMul ((η a) a✝³) ((η a) a✝²))
      ⊢ Eq ((κ a) (Inter.inter a✝³ a✝²)) (HMul.hMul ((κ a) a✝³) ((κ a) a✝²))
    -/
    simpa [ha] using h'a
    /-
      🎉 no goals
    -/


alias ⟨IndepSets.congr, _⟩ := indepSets_congr


lemma iIndep_congr (h : κ =ᵐ[μ] η) : iIndep m κ μ ↔ iIndep m η μ :=
  iIndepSets_congr h


alias ⟨iIndep.congr, _⟩ := iIndep_congr


lemma indep_congr {m₁ m₂ : MeasurableSpace Ω} {_mΩ : MeasurableSpace Ω}
    {κ η : Kernel α Ω} (h : κ =ᵐ[μ] η) : Indep m₁ m₂ κ μ ↔ Indep m₁ m₂ η μ :=
  indepSets_congr h


alias ⟨Indep.congr, _⟩ := indep_congr


lemma iIndepSet_congr (h : κ =ᵐ[μ] η) : iIndepSet s κ μ ↔ iIndepSet s η μ :=
  iIndep_congr h


alias ⟨iIndepSet.congr, _⟩ := iIndepSet_congr


lemma indepSet_congr {s t : Set Ω} (h : κ =ᵐ[μ] η) : IndepSet s t κ μ ↔ IndepSet s t η μ :=
  indep_congr h


alias ⟨indepSet.congr, _⟩ := indepSet_congr


lemma iIndepFun_congr {β : ι → Type*} {m : ∀ x : ι, MeasurableSpace (β x)}
    {f : ∀ x : ι, Ω → β x} (h : κ =ᵐ[μ] η) : iIndepFun m f κ μ ↔ iIndepFun m f η μ :=
  iIndep_congr h


alias ⟨iIndepFun.congr, _⟩ := iIndepFun_congr


lemma indepFun_congr {β γ} [MeasurableSpace β] [MeasurableSpace γ]
    {f : Ω → β} {g : Ω → γ} (h : κ =ᵐ[μ] η) : IndepFun f g κ μ ↔ IndepFun f g η μ :=
  indep_congr h


alias ⟨IndepFun.congr, _⟩ := indepFun_congr


lemma iIndepSets.meas_biInter (h : iIndepSets π κ μ) (s : Finset ι)
    {f : ι → Set Ω} (hf : ∀ i, i ∈ s → f i ∈ π i) :
    ∀ᵐ a ∂μ, κ a (⋂ i ∈ s, f i) = ∏ i ∈ s, κ a (f i) := h s hf


lemma iIndepSets.ae_isProbabilityMeasure (h : iIndepSets π κ μ) :
    ∀ᵐ a ∂μ, IsProbabilityMeasure (κ a) := by
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    π : ι → Set (Set Ω)
    h : ProbabilityTheory.Kernel.iIndepSets π κ μ
    ⊢ Filter.Eventually (fun a => MeasureTheory.IsProbabilityMeasure (κ a)) (Measu …
  -/
  filter_upwards [h.meas_biInter ∅ (f := fun _ ↦ Set.univ) (by simp)] with a ha
  /-
    case h
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    π : ι → Set (Set Ω)
    h : ProbabilityTheory.Kernel.iIndepSets π κ μ
    a : α
    ha : Eq ((κ a) (Set.iInter fun i => Set.iInter fun h => Set.univ)) (EmptyColle …
    ⊢ MeasureTheory.IsProbabilityMeasure (κ a)
  -/
  exact ⟨by simpa using ha⟩
  /-
    🎉 no goals
  -/


lemma iIndepSets.meas_iInter [Fintype ι] (h : iIndepSets π κ μ) (hs : ∀ i, s i ∈ π i) :
    ∀ᵐ a ∂μ, κ a (⋂ i, s i) = ∏ i, κ a (s i) := by
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    π : ι → Set (Set Ω)
    s : ι → Set Ω
    inst✝ : Fintype ι
    h : ProbabilityTheory.Kernel.iIndepSets π κ μ
    hs : ∀ (i : ι), Membership.mem (π i) (s i)
    ⊢ Filter.Eventually (fun a => Eq ((κ a) (Set.iInter fun i => s i)) (Finset.uni …
  -/
  filter_upwards [h.meas_biInter Finset.univ (fun _i _ ↦ hs _)] with a ha using by simp [← ha]
  /-
    🎉 no goals
  -/


lemma iIndep.iIndepSets' (hμ : iIndep m κ μ) :
    iIndepSets (fun x ↦ {s | MeasurableSet[m x] s}) κ μ := hμ


lemma iIndep.ae_isProbabilityMeasure (h : iIndep m κ μ) :
    ∀ᵐ a ∂μ, IsProbabilityMeasure (κ a) :=
  h.iIndepSets'.ae_isProbabilityMeasure


lemma iIndep.meas_biInter (hμ : iIndep m κ μ) (hs : ∀ i, i ∈ S → MeasurableSet[m i] (s i)) :
    ∀ᵐ a ∂μ, κ a (⋂ i ∈ S, s i) = ∏ i ∈ S, κ a (s i) := hμ _ hs


lemma iIndep.meas_iInter [Fintype ι] (h : iIndep m κ μ) (hs : ∀ i, MeasurableSet[m i] (s i)) :
    ∀ᵐ a ∂μ, κ a (⋂ i, s i) = ∏ i, κ a (s i) := by
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    m : ι → MeasurableSpace Ω
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    s : ι → Set Ω
    inst✝ : Fintype ι
    h : ProbabilityTheory.Kernel.iIndep m κ μ
    hs : ∀ (i : ι), MeasurableSet (s i)
    ⊢ Filter.Eventually (fun a => Eq ((κ a) (Set.iInter fun i => s i)) (Finset.uni …
  -/
  filter_upwards [h.meas_biInter (fun i (_ : i ∈ Finset.univ) ↦ hs _)] with a ha
  /-
    case h
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    m : ι → MeasurableSpace Ω
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    s : ι → Set Ω
    inst✝ : Fintype ι
    h : ProbabilityTheory.Kernel.iIndep m κ μ
    hs : ∀ (i : ι), MeasurableSet (s i)
    a : α
    ha : Eq ((κ a) (Set.iInter fun i => Set.iInter fun h => s i)) (Finset.univ.pro …
    ⊢ Eq ((κ a) (Set.iInter fun i => s i)) (Finset.univ.prod fun i => (κ a) (s i))
  -/
  simp [← ha]
  /-
    🎉 no goals
  -/


protected lemma iIndepFun.iIndep (hf : iIndepFun mβ f κ μ) :
    iIndep (fun x ↦ (mβ x).comap (f x)) κ μ := hf


lemma iIndepFun.ae_isProbabilityMeasure (h : iIndepFun mβ f κ μ) :
    ∀ᵐ a ∂μ, IsProbabilityMeasure (κ a) :=
  h.iIndep.ae_isProbabilityMeasure


lemma iIndepFun.meas_biInter (hf : iIndepFun mβ f κ μ)
    (hs : ∀ i, i ∈ S → MeasurableSet[(mβ i).comap (f i)] (s i)) :
    ∀ᵐ a ∂μ, κ a (⋂ i ∈ S, s i) = ∏ i ∈ S, κ a (s i) := hf.iIndep.meas_biInter hs


lemma iIndepFun.meas_iInter [Fintype ι] (hf : iIndepFun mβ f κ μ)
    (hs : ∀ i, MeasurableSet[(mβ i).comap (f i)] (s i)) :
    ∀ᵐ a ∂μ, κ a (⋂ i, s i) = ∏ i, κ a (s i) := hf.iIndep.meas_iInter hs


lemma IndepFun.meas_inter {β γ : Type*} [mβ : MeasurableSpace β] [mγ : MeasurableSpace γ]
    {f : Ω → β} {g : Ω → γ} (hfg : IndepFun f g κ μ)
    {s t : Set Ω} (hs : MeasurableSet[mβ.comap f] s) (ht : MeasurableSet[mγ.comap g] t) :
    ∀ᵐ a ∂μ, κ a (s ∩ t) = κ a s * κ a t := hfg _ _ hs ht


@[symm]
theorem IndepSets.symm {_mΩ : MeasurableSpace Ω} {κ : Kernel α Ω} {μ : Measure α}
    {s₁ s₂ : Set (Set Ω)} (h : IndepSets s₁ s₂ κ μ) :
    IndepSets s₂ s₁ κ μ := by
  /-
    α : Type u_1
    Ω : Type u_2
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    s₁ s₂ : Set (Set Ω)
    h : ProbabilityTheory.Kernel.IndepSets s₁ s₂ κ μ
    ⊢ ProbabilityTheory.Kernel.IndepSets s₂ s₁ κ μ
  -/
  intros t1 t2 ht1 ht2
  /-
    α : Type u_1
    Ω : Type u_2
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    s₁ s₂ : Set (Set Ω)
    h : ProbabilityTheory.Kernel.IndepSets s₁ s₂ κ μ
    t1 t2 : Set Ω
    ht1 : Membership.mem s₂ t1
    ht2 : Membership.mem s₁ t2
    ⊢ Filter.Eventually (fun a => Eq ((κ a) (Inter.inter t1 t2)) (HMul.hMul ((κ a) …
  -/
  filter_upwards [h t2 t1 ht2 ht1] with a ha
  /-
    case h
    α : Type u_1
    Ω : Type u_2
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    s₁ s₂ : Set (Set Ω)
    h : ProbabilityTheory.Kernel.IndepSets s₁ s₂ κ μ
    t1 t2 : Set Ω
    ht1 : Membership.mem s₂ t1
    ht2 : Membership.mem s₁ t2
    a : α
    ha : Eq ((κ a) (Inter.inter t2 t1)) (HMul.hMul ((κ a) t2) ((κ a) t1))
    ⊢ Eq ((κ a) (Inter.inter t1 t2)) (HMul.hMul ((κ a) t1) ((κ a) t2))
  -/
  rwa [Set.inter_comm, mul_comm]
  /-
    🎉 no goals
  -/


@[symm]
theorem Indep.symm {m₁ m₂ : MeasurableSpace Ω} {_mΩ : MeasurableSpace Ω} {κ : Kernel α Ω}
    {μ : Measure α} (h : Indep m₁ m₂ κ μ) :
    Indep m₂ m₁ κ μ :=
  IndepSets.symm h


theorem indep_bot_right (m' : MeasurableSpace Ω) {_mΩ : MeasurableSpace Ω}
    {κ : Kernel α Ω} {μ : Measure α} [IsZeroOrMarkovKernel κ] :
    Indep m' ⊥ κ μ := by
  /-
    α : Type u_1
    Ω : Type u_2
    _mα : MeasurableSpace α
    m' _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel κ
    ⊢ ProbabilityTheory.Kernel.Indep m' Bot.bot κ μ
  -/
  intros s t _ ht
  /-
    α : Type u_1
    Ω : Type u_2
    _mα : MeasurableSpace α
    m' _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel κ
    s t : Set Ω
    a✝ : Membership.mem (setOf fun s => MeasurableSet s) s
    ht : Membership.mem (setOf fun s => MeasurableSet s) t
    ⊢ Filter.Eventually (fun a => Eq ((κ a) (Inter.inter s t)) (HMul.hMul ((κ a) s …
  -/
  rw [Set.mem_setOf_eq, MeasurableSpace.measurableSet_bot_iff] at ht
  /-
    α : Type u_1
    Ω : Type u_2
    _mα : MeasurableSpace α
    m' _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel κ
    s t : Set Ω
    a✝ : Membership.mem (setOf fun s => MeasurableSet s) s
    ht : Or (Eq t EmptyCollection.emptyCollection) (Eq t Set.univ)
    ⊢ Filter.Eventually (fun a => Eq ((κ a) (Inter.inter s t)) (HMul.hMul ((κ a) s …
  -/
  rcases eq_zero_or_isMarkovKernel κ with rfl| h
    /-
      case inl
      α : Type u_1
      Ω : Type u_2
      _mα : MeasurableSpace α
      m' _mΩ : MeasurableSpace Ω
      μ : MeasureTheory.Measure α
      s t : Set Ω
      a✝ : Membership.mem (setOf fun s => MeasurableSet s) s
      ht : Or (Eq t EmptyCollection.emptyCollection) (Eq t Set.univ)
      inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel 0
      ⊢ Filter.Eventually (fun a => Eq ((0 a) (Inter.inter s t)) (HMul.hMul ((0 a) s …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    Ω : Type u_2
    _mα : MeasurableSpace α
    m' _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel κ
    s t : Set Ω
    a✝ : Membership.mem (setOf fun s => MeasurableSet s) s
    ht : Or (Eq t EmptyCollection.emptyCollection) (Eq t Set.univ)
    h : ProbabilityTheory.IsMarkovKernel κ
    ⊢ Filter.Eventually (fun a => Eq ((κ a) (Inter.inter s t)) (HMul.hMul ((κ a) s …
  -/
  refine Filter.Eventually.of_forall (fun a ↦ ?_)
  /-
    case inr
    α : Type u_1
    Ω : Type u_2
    _mα : MeasurableSpace α
    m' _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel κ
    s t : Set Ω
    a✝ : Membership.mem (setOf fun s => MeasurableSet s) s
    ht : Or (Eq t EmptyCollection.emptyCollection) (Eq t Set.univ)
    h : ProbabilityTheory.IsMarkovKernel κ
    a : α
    ⊢ Eq ((κ a) (Inter.inter s t)) (HMul.hMul ((κ a) s) ((κ a) t))
  -/
  cases' ht with ht ht
    /-
      case inr.inl
      α : Type u_1
      Ω : Type u_2
      _mα : MeasurableSpace α
      m' _mΩ : MeasurableSpace Ω
      κ : ProbabilityTheory.Kernel α Ω
      μ : MeasureTheory.Measure α
      inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel κ
      s t : Set Ω
      a✝ : Membership.mem (setOf fun s => MeasurableSet s) s
      h : ProbabilityTheory.IsMarkovKernel κ
      a : α
      ht : Eq t EmptyCollection.emptyCollection
      ⊢ Eq ((κ a) (Inter.inter s t)) (HMul.hMul ((κ a) s) ((κ a) t))
    -/
  · rw [ht, Set.inter_empty, measure_empty, mul_zero]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      α : Type u_1
      Ω : Type u_2
      _mα : MeasurableSpace α
      m' _mΩ : MeasurableSpace Ω
      κ : ProbabilityTheory.Kernel α Ω
      μ : MeasureTheory.Measure α
      inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel κ
      s t : Set Ω
      a✝ : Membership.mem (setOf fun s => MeasurableSet s) s
      h : ProbabilityTheory.IsMarkovKernel κ
      a : α
      ht : Eq t Set.univ
      ⊢ Eq ((κ a) (Inter.inter s t)) (HMul.hMul ((κ a) s) ((κ a) t))
    -/
  · rw [ht, Set.inter_univ, measure_univ, mul_one]
    /-
      🎉 no goals
    -/


theorem indep_bot_left (m' : MeasurableSpace Ω) {_mΩ : MeasurableSpace Ω}
    {κ : Kernel α Ω} {μ : Measure α} [IsZeroOrMarkovKernel κ] :
    Indep ⊥ m' κ μ := (indep_bot_right m').symm


theorem indepSet_empty_right {_mΩ : MeasurableSpace Ω}
    {κ : Kernel α Ω} {μ : Measure α} [IsZeroOrMarkovKernel κ] (s : Set Ω) :
    IndepSet s ∅ κ μ := by
  /-
    α : Type u_1
    Ω : Type u_2
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel κ
    s : Set Ω
    ⊢ ProbabilityTheory.Kernel.IndepSet s EmptyCollection.emptyCollection κ μ
  -/
  simp only [IndepSet, generateFrom_singleton_empty]
  /-
    α : Type u_1
    Ω : Type u_2
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel κ
    s : Set Ω
    ⊢ ProbabilityTheory.Kernel.Indep (MeasurableSpace.generateFrom (Singleton.sing …
  -/
  exact indep_bot_right _
  /-
    🎉 no goals
  -/


theorem indepSet_empty_left {_mΩ : MeasurableSpace Ω} {κ : Kernel α Ω}
    {μ : Measure α} [IsZeroOrMarkovKernel κ] (s : Set Ω) :
    IndepSet ∅ s κ μ :=
  (indepSet_empty_right s).symm


theorem indepSets_of_indepSets_of_le_left {s₁ s₂ s₃ : Set (Set Ω)} {_mΩ : MeasurableSpace Ω}
    {κ : Kernel α Ω} {μ : Measure α} (h_indep : IndepSets s₁ s₂ κ μ) (h31 : s₃ ⊆ s₁) :
    IndepSets s₃ s₂ κ μ :=
  fun t1 t2 ht1 ht2 => h_indep t1 t2 (Set.mem_of_subset_of_mem h31 ht1) ht2


theorem indepSets_of_indepSets_of_le_right {s₁ s₂ s₃ : Set (Set Ω)} {_mΩ : MeasurableSpace Ω}
    {κ : Kernel α Ω} {μ : Measure α} (h_indep : IndepSets s₁ s₂ κ μ) (h32 : s₃ ⊆ s₂) :
    IndepSets s₁ s₃ κ μ :=
  fun t1 t2 ht1 ht2 => h_indep t1 t2 ht1 (Set.mem_of_subset_of_mem h32 ht2)


theorem indep_of_indep_of_le_left {m₁ m₂ m₃ : MeasurableSpace Ω} {_mΩ : MeasurableSpace Ω}
    {κ : Kernel α Ω} {μ : Measure α} (h_indep : Indep m₁ m₂ κ μ) (h31 : m₃ ≤ m₁) :
    Indep m₃ m₂ κ μ :=
  fun t1 t2 ht1 ht2 => h_indep t1 t2 (h31 _ ht1) ht2


theorem indep_of_indep_of_le_right {m₁ m₂ m₃ : MeasurableSpace Ω} {_mΩ : MeasurableSpace Ω}
    {κ : Kernel α Ω} {μ : Measure α} (h_indep : Indep m₁ m₂ κ μ) (h32 : m₃ ≤ m₂) :
    Indep m₁ m₃ κ μ :=
  fun t1 t2 ht1 ht2 => h_indep t1 t2 ht1 (h32 _ ht2)


theorem IndepSets.union {s₁ s₂ s' : Set (Set Ω)} {_mΩ : MeasurableSpace Ω}
    {κ : Kernel α Ω} {μ : Measure α}
    (h₁ : IndepSets s₁ s' κ μ) (h₂ : IndepSets s₂ s' κ μ) :
    IndepSets (s₁ ∪ s₂) s' κ μ := by
  /-
    α : Type u_1
    Ω : Type u_2
    _mα : MeasurableSpace α
    s₁ s₂ s' : Set (Set Ω)
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    h₁ : ProbabilityTheory.Kernel.IndepSets s₁ s' κ μ
    h₂ : ProbabilityTheory.Kernel.IndepSets s₂ s' κ μ
    ⊢ ProbabilityTheory.Kernel.IndepSets (Union.union s₁ s₂) s' κ μ
  -/
  intro t1 t2 ht1 ht2
  /-
    α : Type u_1
    Ω : Type u_2
    _mα : MeasurableSpace α
    s₁ s₂ s' : Set (Set Ω)
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    h₁ : ProbabilityTheory.Kernel.IndepSets s₁ s' κ μ
    h₂ : ProbabilityTheory.Kernel.IndepSets s₂ s' κ μ
    t1 t2 : Set Ω
    ht1 : Membership.mem (Union.union s₁ s₂) t1
    ht2 : Membership.mem s' t2
    ⊢ Filter.Eventually (fun a => Eq ((κ a) (Inter.inter t1 t2)) (HMul.hMul ((κ a) …
  -/
  cases' (Set.mem_union _ _ _).mp ht1 with ht1₁ ht1₂
    /-
      case inl
      α : Type u_1
      Ω : Type u_2
      _mα : MeasurableSpace α
      s₁ s₂ s' : Set (Set Ω)
      _mΩ : MeasurableSpace Ω
      κ : ProbabilityTheory.Kernel α Ω
      μ : MeasureTheory.Measure α
      h₁ : ProbabilityTheory.Kernel.IndepSets s₁ s' κ μ
      h₂ : ProbabilityTheory.Kernel.IndepSets s₂ s' κ μ
      t1 t2 : Set Ω
      ht1 : Membership.mem (Union.union s₁ s₂) t1
      ht2 : Membership.mem s' t2
      ht1₁ : Membership.mem s₁ t1
      ⊢ Filter.Eventually (fun a => Eq ((κ a) (Inter.inter t1 t2)) (HMul.hMul ((κ a) …
    -/
  · exact h₁ t1 t2 ht1₁ ht2
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      Ω : Type u_2
      _mα : MeasurableSpace α
      s₁ s₂ s' : Set (Set Ω)
      _mΩ : MeasurableSpace Ω
      κ : ProbabilityTheory.Kernel α Ω
      μ : MeasureTheory.Measure α
      h₁ : ProbabilityTheory.Kernel.IndepSets s₁ s' κ μ
      h₂ : ProbabilityTheory.Kernel.IndepSets s₂ s' κ μ
      t1 t2 : Set Ω
      ht1 : Membership.mem (Union.union s₁ s₂) t1
      ht2 : Membership.mem s' t2
      ht1₂ : Membership.mem s₂ t1
      ⊢ Filter.Eventually (fun a => Eq ((κ a) (Inter.inter t1 t2)) (HMul.hMul ((κ a) …
    -/
  · exact h₂ t1 t2 ht1₂ ht2
    /-
      🎉 no goals
    -/


@[simp]
theorem IndepSets.union_iff {s₁ s₂ s' : Set (Set Ω)} {_mΩ : MeasurableSpace Ω}
    {κ : Kernel α Ω} {μ : Measure α} :
    IndepSets (s₁ ∪ s₂) s' κ μ ↔ IndepSets s₁ s' κ μ ∧ IndepSets s₂ s' κ μ :=
  ⟨fun h =>
    ⟨indepSets_of_indepSets_of_le_left h Set.subset_union_left,
      indepSets_of_indepSets_of_le_left h Set.subset_union_right⟩,
    fun h => IndepSets.union h.left h.right⟩


theorem IndepSets.iUnion {s : ι → Set (Set Ω)} {s' : Set (Set Ω)} {_mΩ : MeasurableSpace Ω}
    {κ : Kernel α Ω} {μ : Measure α} (hyp : ∀ n, IndepSets (s n) s' κ μ) :
    IndepSets (⋃ n, s n) s' κ μ := by
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    s : ι → Set (Set Ω)
    s' : Set (Set Ω)
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    hyp : ∀ (n : ι), ProbabilityTheory.Kernel.IndepSets (s n) s' κ μ
    ⊢ ProbabilityTheory.Kernel.IndepSets (Set.iUnion fun n => s n) s' κ μ
  -/
  intro t1 t2 ht1 ht2
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    s : ι → Set (Set Ω)
    s' : Set (Set Ω)
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    hyp : ∀ (n : ι), ProbabilityTheory.Kernel.IndepSets (s n) s' κ μ
    t1 t2 : Set Ω
    ht1 : Membership.mem (Set.iUnion fun n => s n) t1
    ht2 : Membership.mem s' t2
    ⊢ Filter.Eventually (fun a => Eq ((κ a) (Inter.inter t1 t2)) (HMul.hMul ((κ a) …
  -/
  rw [Set.mem_iUnion] at ht1
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    s : ι → Set (Set Ω)
    s' : Set (Set Ω)
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    hyp : ∀ (n : ι), ProbabilityTheory.Kernel.IndepSets (s n) s' κ μ
    t1 t2 : Set Ω
    ht1 : Exists fun i => Membership.mem (s i) t1
    ht2 : Membership.mem s' t2
    ⊢ Filter.Eventually (fun a => Eq ((κ a) (Inter.inter t1 t2)) (HMul.hMul ((κ a) …
  -/
  cases' ht1 with n ht1
  /-
    case intro
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    s : ι → Set (Set Ω)
    s' : Set (Set Ω)
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    hyp : ∀ (n : ι), ProbabilityTheory.Kernel.IndepSets (s n) s' κ μ
    t1 t2 : Set Ω
    ht2 : Membership.mem s' t2
    n : ι
    ht1 : Membership.mem (s n) t1
    ⊢ Filter.Eventually (fun a => Eq ((κ a) (Inter.inter t1 t2)) (HMul.hMul ((κ a) …
  -/
  exact hyp n t1 t2 ht1 ht2
  /-
    🎉 no goals
  -/


theorem IndepSets.bUnion {s : ι → Set (Set Ω)} {s' : Set (Set Ω)} {_mΩ : MeasurableSpace Ω}
    {κ : Kernel α Ω} {μ : Measure α} {u : Set ι} (hyp : ∀ n ∈ u, IndepSets (s n) s' κ μ) :
    IndepSets (⋃ n ∈ u, s n) s' κ μ := by
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    s : ι → Set (Set Ω)
    s' : Set (Set Ω)
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    u : Set ι
    hyp : ∀ (n : ι), Membership.mem u n → ProbabilityTheory.Kernel.IndepSets (s n) …
    ⊢ ProbabilityTheory.Kernel.IndepSets (Set.iUnion fun n => Set.iUnion fun h =>  …
  -/
  intro t1 t2 ht1 ht2
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    s : ι → Set (Set Ω)
    s' : Set (Set Ω)
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    u : Set ι
    hyp : ∀ (n : ι), Membership.mem u n → ProbabilityTheory.Kernel.IndepSets (s n) …
    t1 t2 : Set Ω
    ht1 : Membership.mem (Set.iUnion fun n => Set.iUnion fun h => s n) t1
    ht2 : Membership.mem s' t2
    ⊢ Filter.Eventually (fun a => Eq ((κ a) (Inter.inter t1 t2)) (HMul.hMul ((κ a) …
  -/
  simp_rw [Set.mem_iUnion] at ht1
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    s : ι → Set (Set Ω)
    s' : Set (Set Ω)
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    u : Set ι
    hyp : ∀ (n : ι), Membership.mem u n → ProbabilityTheory.Kernel.IndepSets (s n) …
    t1 t2 : Set Ω
    ht2 : Membership.mem s' t2
    ht1 : Exists fun i => Exists fun i_1 => Membership.mem (s i) t1
    ⊢ Filter.Eventually (fun a => Eq ((κ a) (Inter.inter t1 t2)) (HMul.hMul ((κ a) …
  -/
  rcases ht1 with ⟨n, hpn, ht1⟩
  /-
    case intro.intro
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    s : ι → Set (Set Ω)
    s' : Set (Set Ω)
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    u : Set ι
    hyp : ∀ (n : ι), Membership.mem u n → ProbabilityTheory.Kernel.IndepSets (s n) …
    t1 t2 : Set Ω
    ht2 : Membership.mem s' t2
    n : ι
    hpn : Membership.mem u n
    ht1 : Membership.mem (s n) t1
    ⊢ Filter.Eventually (fun a => Eq ((κ a) (Inter.inter t1 t2)) (HMul.hMul ((κ a) …
  -/
  exact hyp n hpn t1 t2 ht1 ht2
  /-
    🎉 no goals
  -/


theorem IndepSets.inter {s₁ s' : Set (Set Ω)} (s₂ : Set (Set Ω)) {_mΩ : MeasurableSpace Ω}
    {κ : Kernel α Ω} {μ : Measure α} (h₁ : IndepSets s₁ s' κ μ) :
    IndepSets (s₁ ∩ s₂) s' κ μ :=
  fun t1 t2 ht1 ht2 => h₁ t1 t2 ((Set.mem_inter_iff _ _ _).mp ht1).left ht2


theorem IndepSets.iInter {s : ι → Set (Set Ω)} {s' : Set (Set Ω)} {_mΩ : MeasurableSpace Ω}
    {κ : Kernel α Ω} {μ : Measure α} (h : ∃ n, IndepSets (s n) s' κ μ) :
    IndepSets (⋂ n, s n) s' κ μ := by
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    s : ι → Set (Set Ω)
    s' : Set (Set Ω)
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    h : Exists fun n => ProbabilityTheory.Kernel.IndepSets (s n) s' κ μ
    ⊢ ProbabilityTheory.Kernel.IndepSets (Set.iInter fun n => s n) s' κ μ
  -/
  intro t1 t2 ht1 ht2; cases' h with n h; exact h t1 t2 (Set.mem_iInter.mp ht1 n) ht2
                                          /-
                                            🎉 no goals
                                          -/


theorem IndepSets.bInter {s : ι → Set (Set Ω)} {s' : Set (Set Ω)} {_mΩ : MeasurableSpace Ω}
    {κ : Kernel α Ω} {μ : Measure α} {u : Set ι} (h : ∃ n ∈ u, IndepSets (s n) s' κ μ) :
    IndepSets (⋂ n ∈ u, s n) s' κ μ := by
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    s : ι → Set (Set Ω)
    s' : Set (Set Ω)
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    u : Set ι
    h : Exists fun n => And (Membership.mem u n) (ProbabilityTheory.Kernel.IndepSe …
    ⊢ ProbabilityTheory.Kernel.IndepSets (Set.iInter fun n => Set.iInter fun h =>  …
  -/
  intro t1 t2 ht1 ht2
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    s : ι → Set (Set Ω)
    s' : Set (Set Ω)
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    u : Set ι
    h : Exists fun n => And (Membership.mem u n) (ProbabilityTheory.Kernel.IndepSe …
    t1 t2 : Set Ω
    ht1 : Membership.mem (Set.iInter fun n => Set.iInter fun h => s n) t1
    ht2 : Membership.mem s' t2
    ⊢ Filter.Eventually (fun a => Eq ((κ a) (Inter.inter t1 t2)) (HMul.hMul ((κ a) …
  -/
  rcases h with ⟨n, hn, h⟩
  /-
    case intro.intro
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    s : ι → Set (Set Ω)
    s' : Set (Set Ω)
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    u : Set ι
    t1 t2 : Set Ω
    ht1 : Membership.mem (Set.iInter fun n => Set.iInter fun h => s n) t1
    ht2 : Membership.mem s' t2
    n : ι
    hn : Membership.mem u n
    h : ProbabilityTheory.Kernel.IndepSets (s n) s' κ μ
    ⊢ Filter.Eventually (fun a => Eq ((κ a) (Inter.inter t1 t2)) (HMul.hMul ((κ a) …
  -/
  exact h t1 t2 (Set.biInter_subset_of_mem hn ht1) ht2
  /-
    🎉 no goals
  -/


theorem iIndep_comap_mem_iff {f : ι → Set Ω} {_mΩ : MeasurableSpace Ω}
    {κ : Kernel α Ω} {μ : Measure α} :
    iIndep (fun i => MeasurableSpace.comap (· ∈ f i) ⊤) κ μ ↔ iIndepSet f κ μ := by
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    f : ι → Set Ω
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    ⊢ Iff (ProbabilityTheory.Kernel.iIndep (fun i => MeasurableSpace.comap (fun x  …
  -/
  simp_rw [← generateFrom_singleton, iIndepSet]
  /-
    🎉 no goals
  -/


theorem iIndepSets_singleton_iff {s : ι → Set Ω} {_mΩ : MeasurableSpace Ω}
    {κ : Kernel α Ω} {μ : Measure α} :
    iIndepSets (fun i ↦ {s i}) κ μ ↔
      ∀ S : Finset ι, ∀ᵐ a ∂μ, κ a (⋂ i ∈ S, s i) = ∏ i ∈ S, κ a (s i) := by
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    s : ι → Set Ω
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    ⊢ Iff (ProbabilityTheory.Kernel.iIndepSets (fun i => Singleton.singleton (s i) …
  -/
  refine ⟨fun h S ↦ h S (fun i _ ↦ rfl), fun h S f hf ↦ ?_⟩
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    s : ι → Set Ω
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    h : ∀ (S : Finset ι), Filter.Eventually (fun a => Eq ((κ a) (Set.iInter fun i  …
    S : Finset ι
    f : ι → Set Ω
    hf : ∀ (i : ι), Membership.mem S i → Membership.mem ((fun i => Singleton.singl …
    ⊢ Filter.Eventually (fun a => Eq ((κ a) (Set.iInter fun i => Set.iInter fun h  …
  -/
  filter_upwards [h S] with a ha
  /-
    case h
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    s : ι → Set Ω
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    h : ∀ (S : Finset ι), Filter.Eventually (fun a => Eq ((κ a) (Set.iInter fun i  …
    S : Finset ι
    f : ι → Set Ω
    hf : ∀ (i : ι), Membership.mem S i → Membership.mem ((fun i => Singleton.singl …
    a : α
    ha : Eq ((κ a) (Set.iInter fun i => Set.iInter fun h => s i)) (S.prod fun i => …
    ⊢ Eq ((κ a) (Set.iInter fun i => Set.iInter fun h => f i)) (S.prod fun i => (κ …
  -/
  have : ∀ i ∈ S, κ a (f i) = κ a (s i) := fun i hi ↦ by rw [hf i hi]
  /-
    case h
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    s : ι → Set Ω
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    h : ∀ (S : Finset ι), Filter.Eventually (fun a => Eq ((κ a) (Set.iInter fun i  …
    S : Finset ι
    f : ι → Set Ω
    hf : ∀ (i : ι), Membership.mem S i → Membership.mem ((fun i => Singleton.singl …
    a : α
    ha : Eq ((κ a) (Set.iInter fun i => Set.iInter fun h => s i)) (S.prod fun i => …
    this : ∀ (i : ι), Membership.mem S i → Eq ((κ a) (f i)) ((κ a) (s i))
    ⊢ Eq ((κ a) (Set.iInter fun i => Set.iInter fun h => f i)) (S.prod fun i => (κ …
  -/
  rwa [Finset.prod_congr rfl this, Set.iInter₂_congr hf]
  /-
    🎉 no goals
  -/


theorem indepSets_singleton_iff {s t : Set Ω} {_mΩ : MeasurableSpace Ω}
    {κ : Kernel α Ω} {μ : Measure α} :
    IndepSets {s} {t} κ μ ↔ ∀ᵐ a ∂μ, κ a (s ∩ t) = κ a s * κ a t :=
  ⟨fun h ↦ h s t rfl rfl,
                            /-
                              α : Type u_1
                              Ω : Type u_2
                              _mα : MeasurableSpace α
                              s t : Set Ω
                              _mΩ : MeasurableSpace Ω
                              κ : ProbabilityTheory.Kernel α Ω
                              μ : MeasureTheory.Measure α
                              h : Filter.Eventually (fun a => Eq ((κ a) (Inter.inter s t)) (HMul.hMul ((κ a) …
                              s1 t1 : Set Ω
                              hs1 : Membership.mem (Singleton.singleton s) s1
                              ht1 : Membership.mem (Singleton.singleton t) t1
                              ⊢ Filter.Eventually (fun a => Eq ((κ a) (Inter.inter s1 t1)) (HMul.hMul ((κ a) …
                            -/
   fun h s1 t1 hs1 ht1 ↦ by rwa [Set.mem_singleton_iff.mp hs1, Set.mem_singleton_iff.mp ht1]⟩
                            /-
                              🎉 no goals
                            -/


theorem iIndepSets.indepSets {s : ι → Set (Set Ω)} {_mΩ : MeasurableSpace Ω}
    {κ : Kernel α Ω} {μ : Measure α} (h_indep : iIndepSets s κ μ) {i j : ι} (hij : i ≠ j) :
    IndepSets (s i) (s j) κ μ := by
  classical
  intro t₁ t₂ ht₁ ht₂
  have hf_m : ∀ x : ι, x ∈ ({i, j} : Finset ι) → ite (x = i) t₁ t₂ ∈ s x := by
    intro x hx
    cases' Finset.mem_insert.mp hx with hx hx
    · simp [hx, ht₁]
    · simp [Finset.mem_singleton.mp hx, hij.symm, ht₂]
  have h1 : t₁ = ite (i = i) t₁ t₂ := by simp only [if_true, eq_self_iff_true]
  have h2 : t₂ = ite (j = i) t₁ t₂ := by simp only [hij.symm, if_false]
  have h_inter : ⋂ (t : ι) (_ : t ∈ ({i, j} : Finset ι)), ite (t = i) t₁ t₂ =
      ite (i = i) t₁ t₂ ∩ ite (j = i) t₁ t₂ := by
    simp only [Finset.set_biInter_singleton, Finset.set_biInter_insert]
  filter_upwards [h_indep {i, j} hf_m] with a h_indep'
  have h_prod : (∏ t ∈ ({i, j} : Finset ι), κ a (ite (t = i) t₁ t₂))
      = κ a (ite (i = i) t₁ t₂) * κ a (ite (j = i) t₁ t₂) := by
    simp only [hij, Finset.prod_singleton, Finset.prod_insert, not_false_iff,
      Finset.mem_singleton]
  rw [h1]
  nth_rw 2 [h2]
  nth_rw 4 [h2]
  rw [← h_inter, ← h_prod, h_indep']


theorem iIndep.indep {m : ι → MeasurableSpace Ω} {_mΩ : MeasurableSpace Ω}
    {κ : Kernel α Ω} {μ : Measure α}
    (h_indep : iIndep m κ μ) {i j : ι} (hij : i ≠ j) : Indep (m i) (m j) κ μ :=
  iIndepSets.indepSets h_indep hij


theorem iIndepFun.indepFun {_mΩ : MeasurableSpace Ω}
    {κ : Kernel α Ω} {μ : Measure α} {β : ι → Type*}
    {m : ∀ x, MeasurableSpace (β x)} {f : ∀ i, Ω → β i} (hf_Indep : iIndepFun m f κ μ) {i j : ι}
    (hij : i ≠ j) : IndepFun (f i) (f j) κ μ :=
  hf_Indep.indep hij


theorem iIndep.iIndepSets {_mΩ : MeasurableSpace Ω}
    {κ : Kernel α Ω} {μ : Measure α} {m : ι → MeasurableSpace Ω}
    {s : ι → Set (Set Ω)} (hms : ∀ n, m n = generateFrom (s n)) (h_indep : iIndep m κ μ) :
    iIndepSets s κ μ :=
  fun S f hfs =>
  h_indep S fun x hxS =>
    ((hms x).symm ▸ measurableSet_generateFrom (hfs x hxS) : MeasurableSet[m x] (f x))


theorem Indep.indepSets {_mΩ : MeasurableSpace Ω}
    {κ : Kernel α Ω} {μ : Measure α} {s1 s2 : Set (Set Ω)}
    (h_indep : Indep (generateFrom s1) (generateFrom s2) κ μ) :
    IndepSets s1 s2 κ μ :=
  fun t1 t2 ht1 ht2 =>
  h_indep t1 t2 (measurableSet_generateFrom ht1) (measurableSet_generateFrom ht2)


theorem IndepSets.indep_aux {m₂ m : MeasurableSpace Ω}
    {κ : Kernel α Ω} {μ : Measure α} [IsZeroOrMarkovKernel κ] {p1 p2 : Set (Set Ω)} (h2 : m₂ ≤ m)
    (hp2 : IsPiSystem p2) (hpm2 : m₂ = generateFrom p2) (hyp : IndepSets p1 p2 κ μ) {t1 t2 : Set Ω}
    (ht1 : t1 ∈ p1) (ht1m : MeasurableSet[m] t1) (ht2m : MeasurableSet[m₂] t2) :
    ∀ᵐ a ∂μ, κ a (t1 ∩ t2) = κ a t1 * κ a t2 := by
  /-
    α : Type u_1
    Ω : Type u_2
    _mα : MeasurableSpace α
    m₂ m : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel κ
    p1 p2 : Set (Set Ω)
    h2 : LE.le m₂ m
    hp2 : IsPiSystem p2
    hpm2 : Eq m₂ (MeasurableSpace.generateFrom p2)
    hyp : ProbabilityTheory.Kernel.IndepSets p1 p2 κ μ
    t1 t2 : Set Ω
    ht1 : Membership.mem p1 t1
    ht1m : MeasurableSet t1
    ht2m : MeasurableSet t2
    ⊢ Filter.Eventually (fun a => Eq ((κ a) (Inter.inter t1 t2)) (HMul.hMul ((κ a) …
  -/
  rcases eq_zero_or_isMarkovKernel κ with rfl | h
    /-
      case inl
      α : Type u_1
      Ω : Type u_2
      _mα : MeasurableSpace α
      m₂ m : MeasurableSpace Ω
      μ : MeasureTheory.Measure α
      p1 p2 : Set (Set Ω)
      h2 : LE.le m₂ m
      hp2 : IsPiSystem p2
      hpm2 : Eq m₂ (MeasurableSpace.generateFrom p2)
      t1 t2 : Set Ω
      ht1 : Membership.mem p1 t1
      ht1m : MeasurableSet t1
      ht2m : MeasurableSet t2
      inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel 0
      hyp : ProbabilityTheory.Kernel.IndepSets p1 p2 0 μ
      ⊢ Filter.Eventually (fun a => Eq ((0 a) (Inter.inter t1 t2)) (HMul.hMul ((0 a) …
    -/
  · simp
    /-
      🎉 no goals
    -/
  induction t2, ht2m using induction_on_inter hpm2 hp2 with
  | empty => simp
  | basic u hu => exact hyp t1 u ht1 hu
  | compl u hu ihu =>
    filter_upwards [ihu] with a ha
    rw [← Set.diff_eq, ← Set.diff_self_inter,
      measure_diff inter_subset_left (ht1m.inter (h2 _ hu)).nullMeasurableSet (measure_ne_top _ _),
      ha, measure_compl (h2 _ hu) (measure_ne_top _ _), measure_univ, ENNReal.mul_sub, mul_one]
    exact fun _ _ ↦ measure_ne_top _ _
  | iUnion f hfd hfm ihf =>
    rw [← ae_all_iff] at ihf
    filter_upwards [ihf] with a ha
    rw [inter_iUnion, measure_iUnion, measure_iUnion hfd fun i ↦ h2 _ (hfm i)]
    · simp only [ENNReal.tsum_mul_left, ha]
    · exact hfd.mono fun i j h ↦ (h.inter_left' _).inter_right' _
    · exact fun i ↦ .inter ht1m (h2 _ <| hfm i)


/-- The measurable space structures generated by independent pi-systems are independent. -/
theorem IndepSets.indep {m1 m2 m : MeasurableSpace Ω} {κ : Kernel α Ω} {μ : Measure α}
    [IsZeroOrMarkovKernel κ] {p1 p2 : Set (Set Ω)} (h1 : m1 ≤ m) (h2 : m2 ≤ m) (hp1 : IsPiSystem p1)
    (hp2 : IsPiSystem p2) (hpm1 : m1 = generateFrom p1) (hpm2 : m2 = generateFrom p2)
    (hyp : IndepSets p1 p2 κ μ) :
    Indep m1 m2 κ μ := by
  /-
    α : Type u_1
    Ω : Type u_2
    _mα : MeasurableSpace α
    m1 m2 m : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel κ
    p1 p2 : Set (Set Ω)
    h1 : LE.le m1 m
    h2 : LE.le m2 m
    hp1 : IsPiSystem p1
    hp2 : IsPiSystem p2
    hpm1 : Eq m1 (MeasurableSpace.generateFrom p1)
    hpm2 : Eq m2 (MeasurableSpace.generateFrom p2)
    hyp : ProbabilityTheory.Kernel.IndepSets p1 p2 κ μ
    ⊢ ProbabilityTheory.Kernel.Indep m1 m2 κ μ
  -/
  rcases eq_zero_or_isMarkovKernel κ with rfl | h
    /-
      case inl
      α : Type u_1
      Ω : Type u_2
      _mα : MeasurableSpace α
      m1 m2 m : MeasurableSpace Ω
      μ : MeasureTheory.Measure α
      p1 p2 : Set (Set Ω)
      h1 : LE.le m1 m
      h2 : LE.le m2 m
      hp1 : IsPiSystem p1
      hp2 : IsPiSystem p2
      hpm1 : Eq m1 (MeasurableSpace.generateFrom p1)
      hpm2 : Eq m2 (MeasurableSpace.generateFrom p2)
      inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel 0
      hyp : ProbabilityTheory.Kernel.IndepSets p1 p2 0 μ
      ⊢ ProbabilityTheory.Kernel.Indep m1 m2 0 μ
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    Ω : Type u_2
    _mα : MeasurableSpace α
    m1 m2 m : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel κ
    p1 p2 : Set (Set Ω)
    h1 : LE.le m1 m
    h2 : LE.le m2 m
    hp1 : IsPiSystem p1
    hp2 : IsPiSystem p2
    hpm1 : Eq m1 (MeasurableSpace.generateFrom p1)
    hpm2 : Eq m2 (MeasurableSpace.generateFrom p2)
    hyp : ProbabilityTheory.Kernel.IndepSets p1 p2 κ μ
    h : ProbabilityTheory.IsMarkovKernel κ
    ⊢ ProbabilityTheory.Kernel.Indep m1 m2 κ μ
  -/
  intros t1 t2 ht1 ht2
  induction t1, ht1 using induction_on_inter hpm1 hp1 with
  | empty =>
    simp only [Set.empty_inter, measure_empty, zero_mul, eq_self_iff_true, Filter.eventually_true]
  | basic t ht =>
    refine IndepSets.indep_aux h2 hp2 hpm2 hyp ht (h1 _ ?_) ht2
    rw [hpm1]
    exact measurableSet_generateFrom ht
  | compl t ht iht =>
    filter_upwards [iht] with a ha
    have : tᶜ ∩ t2 = t2 \ (t ∩ t2) := by
      rw [Set.inter_comm t, Set.diff_self_inter, Set.diff_eq_compl_inter]
    rw [this, Set.inter_comm t t2,
      measure_diff Set.inter_subset_left ((h2 _ ht2).inter (h1 _ ht)).nullMeasurableSet
        (measure_ne_top (κ a) _),
      Set.inter_comm, ha, measure_compl (h1 _ ht) (measure_ne_top (κ a) t), measure_univ,
      mul_comm (1 - κ a t), ENNReal.mul_sub (fun _ _ ↦ measure_ne_top (κ a) _), mul_one, mul_comm]
  | iUnion f hf_disj hf_meas h =>
    rw [← ae_all_iff] at h
    filter_upwards [h] with a ha
    rw [Set.inter_comm, Set.inter_iUnion, measure_iUnion]
    · rw [measure_iUnion hf_disj (fun i ↦ h1 _ (hf_meas i))]
      rw [← ENNReal.tsum_mul_right]
      congr 1 with i
      rw [Set.inter_comm t2, ha i]
    · intros i j hij
      rw [Function.onFun, Set.inter_comm t2, Set.inter_comm t2]
      exact Disjoint.inter_left _ (Disjoint.inter_right _ (hf_disj hij))
    · exact fun i ↦ (h2 _ ht2).inter (h1 _ (hf_meas i))


theorem IndepSets.indep' {_mΩ : MeasurableSpace Ω}
    {κ : Kernel α Ω} {μ : Measure α} [IsZeroOrMarkovKernel κ]
    {p1 p2 : Set (Set Ω)} (hp1m : ∀ s ∈ p1, MeasurableSet s) (hp2m : ∀ s ∈ p2, MeasurableSet s)
    (hp1 : IsPiSystem p1) (hp2 : IsPiSystem p2) (hyp : IndepSets p1 p2 κ μ) :
    Indep (generateFrom p1) (generateFrom p2) κ μ :=
  hyp.indep (generateFrom_le hp1m) (generateFrom_le hp2m) hp1 hp2 rfl rfl


theorem indepSets_piiUnionInter_of_disjoint {s : ι → Set (Set Ω)}
    {S T : Set ι} (h_indep : iIndepSets s κ μ) (hST : Disjoint S T) :
    IndepSets (piiUnionInter s S) (piiUnionInter s T) κ μ := by
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    s : ι → Set (Set Ω)
    S T : Set ι
    h_indep : ProbabilityTheory.Kernel.iIndepSets s κ μ
    hST : Disjoint S T
    ⊢ ProbabilityTheory.Kernel.IndepSets (piiUnionInter s S) (piiUnionInter s T) κ μ
  -/
  rintro t1 t2 ⟨p1, hp1, f1, ht1_m, ht1_eq⟩ ⟨p2, hp2, f2, ht2_m, ht2_eq⟩
  classical
  let g i := ite (i ∈ p1) (f1 i) Set.univ ∩ ite (i ∈ p2) (f2 i) Set.univ
  have h_P_inter : ∀ᵐ a ∂μ, κ a (t1 ∩ t2) = ∏ n ∈ p1 ∪ p2, κ a (g n) := by
    have hgm : ∀ i ∈ p1 ∪ p2, g i ∈ s i := by
      intro i hi_mem_union
      rw [Finset.mem_union] at hi_mem_union
      cases' hi_mem_union with hi1 hi2
      · have hi2 : i ∉ p2 := fun hip2 => Set.disjoint_left.mp hST (hp1 hi1) (hp2 hip2)
        simp_rw [g, if_pos hi1, if_neg hi2, Set.inter_univ]
        exact ht1_m i hi1
      · have hi1 : i ∉ p1 := fun hip1 => Set.disjoint_right.mp hST (hp2 hi2) (hp1 hip1)
        simp_rw [g, if_neg hi1, if_pos hi2, Set.univ_inter]
        exact ht2_m i hi2
    have h_p1_inter_p2 :
      ((⋂ x ∈ p1, f1 x) ∩ ⋂ x ∈ p2, f2 x) =
        ⋂ i ∈ p1 ∪ p2, ite (i ∈ p1) (f1 i) Set.univ ∩ ite (i ∈ p2) (f2 i) Set.univ := by
      ext1 x
      simp only [Set.mem_ite_univ_right, Set.mem_inter_iff, Set.mem_iInter, Finset.mem_union]
      exact
        ⟨fun h i _ => ⟨h.1 i, h.2 i⟩, fun h =>
          ⟨fun i hi => (h i (Or.inl hi)).1 hi, fun i hi => (h i (Or.inr hi)).2 hi⟩⟩
    filter_upwards [h_indep _ hgm] with a ha
    rw [ht1_eq, ht2_eq, h_p1_inter_p2, ← ha]
  filter_upwards [h_P_inter, h_indep p1 ht1_m, h_indep p2 ht2_m, h_indep.ae_isProbabilityMeasure]
    with a h_P_inter ha1 ha2 h'
  have h_μg : ∀ n, κ a (g n) = (ite (n ∈ p1) (κ a (f1 n)) 1) * (ite (n ∈ p2) (κ a (f2 n)) 1) := by
    intro n
    dsimp only [g]
    split_ifs with h1 h2
    · exact absurd rfl (Set.disjoint_iff_forall_ne.mp hST (hp1 h1) (hp2 h2))
    all_goals simp only [measure_univ, one_mul, mul_one, Set.inter_univ, Set.univ_inter]
  simp_rw [h_P_inter, h_μg, Finset.prod_mul_distrib,
    Finset.prod_ite_mem (p1 ∪ p2) p1 (fun x ↦ κ a (f1 x)), Finset.union_inter_cancel_left,
    Finset.prod_ite_mem (p1 ∪ p2) p2 (fun x => κ a (f2 x)), Finset.union_inter_cancel_right, ht1_eq,
      ← ha1, ht2_eq, ← ha2]


theorem iIndepSet.indep_generateFrom_of_disjoint {s : ι → Set Ω}
    (hsm : ∀ n, MeasurableSet (s n)) (hs : iIndepSet s κ μ) (S T : Set ι) (hST : Disjoint S T) :
    Indep (generateFrom { t | ∃ n ∈ S, s n = t }) (generateFrom { t | ∃ k ∈ T, s k = t }) κ μ := by
  classical
  rcases eq_or_ne μ 0 with rfl | hμ
  · simp
  obtain ⟨η, η_eq, hη⟩ : ∃ (η : Kernel α Ω), κ =ᵐ[μ] η ∧ IsMarkovKernel η :=
    exists_ae_eq_isMarkovKernel hs.ae_isProbabilityMeasure hμ
  apply Indep.congr (Filter.EventuallyEq.symm η_eq)
  rw [← generateFrom_piiUnionInter_singleton_left, ← generateFrom_piiUnionInter_singleton_left]
  refine
    IndepSets.indep'
      (fun t ht => generateFrom_piiUnionInter_le _ ?_ _ _ (measurableSet_generateFrom ht))
      (fun t ht => generateFrom_piiUnionInter_le _ ?_ _ _ (measurableSet_generateFrom ht)) ?_ ?_ ?_
  · exact fun k => generateFrom_le fun t ht => (Set.mem_singleton_iff.1 ht).symm ▸ hsm k
  · exact fun k => generateFrom_le fun t ht => (Set.mem_singleton_iff.1 ht).symm ▸ hsm k
  · exact isPiSystem_piiUnionInter _ (fun k => IsPiSystem.singleton _) _
  · exact isPiSystem_piiUnionInter _ (fun k => IsPiSystem.singleton _) _
  · exact indepSets_piiUnionInter_of_disjoint (iIndep.iIndepSets (fun n => rfl) (hs.congr η_eq)) hST


theorem indep_iSup_of_disjoint {m : ι → MeasurableSpace Ω}
    (h_le : ∀ i, m i ≤ _mΩ) (h_indep : iIndep m κ μ) {S T : Set ι} (hST : Disjoint S T) :
    Indep (⨆ i ∈ S, m i) (⨆ i ∈ T, m i) κ μ := by
  classical
  rcases eq_or_ne μ 0 with rfl | hμ
  · simp
  obtain ⟨η, η_eq, hη⟩ : ∃ (η : Kernel α Ω), κ =ᵐ[μ] η ∧ IsMarkovKernel η :=
    exists_ae_eq_isMarkovKernel h_indep.ae_isProbabilityMeasure hμ
  apply Indep.congr (Filter.EventuallyEq.symm η_eq)
  refine
    IndepSets.indep (iSup₂_le fun i _ => h_le i) (iSup₂_le fun i _ => h_le i) ?_ ?_
      (generateFrom_piiUnionInter_measurableSet m S).symm
      (generateFrom_piiUnionInter_measurableSet m T).symm ?_
  · exact isPiSystem_piiUnionInter _ (fun n => @isPiSystem_measurableSet Ω (m n)) _
  · exact isPiSystem_piiUnionInter _ (fun n => @isPiSystem_measurableSet Ω (m n)) _
  · exact indepSets_piiUnionInter_of_disjoint (h_indep.congr η_eq) hST


theorem indep_iSup_of_directed_le {Ω} {m : ι → MeasurableSpace Ω} {m' m0 : MeasurableSpace Ω}
    {κ : Kernel α Ω} {μ : Measure α} [IsZeroOrMarkovKernel κ] (h_indep : ∀ i, Indep (m i) m' κ μ)
    (h_le : ∀ i, m i ≤ m0) (h_le' : m' ≤ m0) (hm : Directed (· ≤ ·) m) :
    Indep (⨆ i, m i) m' κ μ := by
  /-
    α : Type u_1
    ι : Type u_3
    _mα : MeasurableSpace α
    Ω : Type u_4
    m : ι → MeasurableSpace Ω
    m' m0 : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel κ
    h_indep : ∀ (i : ι), ProbabilityTheory.Kernel.Indep (m i) m' κ μ
    h_le : ∀ (i : ι), LE.le (m i) m0
    h_le' : LE.le m' m0
    hm : Directed (fun x1 x2 => LE.le x1 x2) m
    ⊢ ProbabilityTheory.Kernel.Indep (iSup fun i => m i) m' κ μ
  -/
  let p : ι → Set (Set Ω) := fun n => { t | MeasurableSet[m n] t }
  /-
    α : Type u_1
    ι : Type u_3
    _mα : MeasurableSpace α
    Ω : Type u_4
    m : ι → MeasurableSpace Ω
    m' m0 : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel κ
    h_indep : ∀ (i : ι), ProbabilityTheory.Kernel.Indep (m i) m' κ μ
    h_le : ∀ (i : ι), LE.le (m i) m0
    h_le' : LE.le m' m0
    hm : Directed (fun x1 x2 => LE.le x1 x2) m
    p : ι → Set (Set Ω) := fun n => setOf fun t => MeasurableSet t
    ⊢ ProbabilityTheory.Kernel.Indep (iSup fun i => m i) m' κ μ
  -/
  have hp : ∀ n, IsPiSystem (p n) := fun n => @isPiSystem_measurableSet Ω (m n)
  have h_gen_n : ∀ n, m n = generateFrom (p n) := fun n =>
    (@generateFrom_measurableSet Ω (m n)).symm
  /-
    α : Type u_1
    ι : Type u_3
    _mα : MeasurableSpace α
    Ω : Type u_4
    m : ι → MeasurableSpace Ω
    m' m0 : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel κ
    h_indep : ∀ (i : ι), ProbabilityTheory.Kernel.Indep (m i) m' κ μ
    h_le : ∀ (i : ι), LE.le (m i) m0
    h_le' : LE.le m' m0
    hm : Directed (fun x1 x2 => LE.le x1 x2) m
    p : ι → Set (Set Ω) := fun n => setOf fun t => MeasurableSet t
    hp : ∀ (n : ι), IsPiSystem (p n)
    h_gen_n : ∀ (n : ι), Eq (m n) (MeasurableSpace.generateFrom (p n))
    ⊢ ProbabilityTheory.Kernel.Indep (iSup fun i => m i) m' κ μ
  -/
  have hp_supr_pi : IsPiSystem (⋃ n, p n) := isPiSystem_iUnion_of_directed_le p hp hm
  /-
    α : Type u_1
    ι : Type u_3
    _mα : MeasurableSpace α
    Ω : Type u_4
    m : ι → MeasurableSpace Ω
    m' m0 : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel κ
    h_indep : ∀ (i : ι), ProbabilityTheory.Kernel.Indep (m i) m' κ μ
    h_le : ∀ (i : ι), LE.le (m i) m0
    h_le' : LE.le m' m0
    hm : Directed (fun x1 x2 => LE.le x1 x2) m
    p : ι → Set (Set Ω) := fun n => setOf fun t => MeasurableSet t
    hp : ∀ (n : ι), IsPiSystem (p n)
    h_gen_n : ∀ (n : ι), Eq (m n) (MeasurableSpace.generateFrom (p n))
    hp_supr_pi : IsPiSystem (Set.iUnion fun n => p n)
    ⊢ ProbabilityTheory.Kernel.Indep (iSup fun i => m i) m' κ μ
  -/
  let p' := { t : Set Ω | MeasurableSet[m'] t }
  /-
    α : Type u_1
    ι : Type u_3
    _mα : MeasurableSpace α
    Ω : Type u_4
    m : ι → MeasurableSpace Ω
    m' m0 : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel κ
    h_indep : ∀ (i : ι), ProbabilityTheory.Kernel.Indep (m i) m' κ μ
    h_le : ∀ (i : ι), LE.le (m i) m0
    h_le' : LE.le m' m0
    hm : Directed (fun x1 x2 => LE.le x1 x2) m
    p : ι → Set (Set Ω) := fun n => setOf fun t => MeasurableSet t
    hp : ∀ (n : ι), IsPiSystem (p n)
    h_gen_n : ∀ (n : ι), Eq (m n) (MeasurableSpace.generateFrom (p n))
    hp_supr_pi : IsPiSystem (Set.iUnion fun n => p n)
    p' : Set (Set Ω) := setOf fun t => MeasurableSet t
    ⊢ ProbabilityTheory.Kernel.Indep (iSup fun i => m i) m' κ μ
  -/
  have hp'_pi : IsPiSystem p' := @isPiSystem_measurableSet Ω m'
  /-
    α : Type u_1
    ι : Type u_3
    _mα : MeasurableSpace α
    Ω : Type u_4
    m : ι → MeasurableSpace Ω
    m' m0 : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel κ
    h_indep : ∀ (i : ι), ProbabilityTheory.Kernel.Indep (m i) m' κ μ
    h_le : ∀ (i : ι), LE.le (m i) m0
    h_le' : LE.le m' m0
    hm : Directed (fun x1 x2 => LE.le x1 x2) m
    p : ι → Set (Set Ω) := fun n => setOf fun t => MeasurableSet t
    hp : ∀ (n : ι), IsPiSystem (p n)
    h_gen_n : ∀ (n : ι), Eq (m n) (MeasurableSpace.generateFrom (p n))
    hp_supr_pi : IsPiSystem (Set.iUnion fun n => p n)
    p' : Set (Set Ω) := setOf fun t => MeasurableSet t
    hp'_pi : IsPiSystem p'
    ⊢ ProbabilityTheory.Kernel.Indep (iSup fun i => m i) m' κ μ
  -/
  have h_gen' : m' = generateFrom p' := (@generateFrom_measurableSet Ω m').symm
  -- the π-systems defined are independent
  have h_pi_system_indep : IndepSets (⋃ n, p n) p' κ μ := by
    refine IndepSets.iUnion ?_
    conv at h_indep =>
      intro i
      rw [h_gen_n i, h_gen']
    exact fun n => (h_indep n).indepSets
  -- now go from π-systems to σ-algebras
  /-
    α : Type u_1
    ι : Type u_3
    _mα : MeasurableSpace α
    Ω : Type u_4
    m : ι → MeasurableSpace Ω
    m' m0 : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel κ
    h_indep : ∀ (i : ι), ProbabilityTheory.Kernel.Indep (m i) m' κ μ
    h_le : ∀ (i : ι), LE.le (m i) m0
    h_le' : LE.le m' m0
    hm : Directed (fun x1 x2 => LE.le x1 x2) m
    p : ι → Set (Set Ω) := fun n => setOf fun t => MeasurableSet t
    hp : ∀ (n : ι), IsPiSystem (p n)
    h_gen_n : ∀ (n : ι), Eq (m n) (MeasurableSpace.generateFrom (p n))
    hp_supr_pi : IsPiSystem (Set.iUnion fun n => p n)
    p' : Set (Set Ω) := setOf fun t => MeasurableSet t
    hp'_pi : IsPiSystem p'
    h_gen' : Eq m' (MeasurableSpace.generateFrom p')
    h_pi_system_indep : ProbabilityTheory.Kernel.IndepSets (Set.iUnion fun n => p  …
    ⊢ ProbabilityTheory.Kernel.Indep (iSup fun i => m i) m' κ μ
  -/
  refine IndepSets.indep (iSup_le h_le) h_le' hp_supr_pi hp'_pi ?_ h_gen' h_pi_system_indep
  /-
    α : Type u_1
    ι : Type u_3
    _mα : MeasurableSpace α
    Ω : Type u_4
    m : ι → MeasurableSpace Ω
    m' m0 : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel κ
    h_indep : ∀ (i : ι), ProbabilityTheory.Kernel.Indep (m i) m' κ μ
    h_le : ∀ (i : ι), LE.le (m i) m0
    h_le' : LE.le m' m0
    hm : Directed (fun x1 x2 => LE.le x1 x2) m
    p : ι → Set (Set Ω) := fun n => setOf fun t => MeasurableSet t
    hp : ∀ (n : ι), IsPiSystem (p n)
    h_gen_n : ∀ (n : ι), Eq (m n) (MeasurableSpace.generateFrom (p n))
    hp_supr_pi : IsPiSystem (Set.iUnion fun n => p n)
    p' : Set (Set Ω) := setOf fun t => MeasurableSet t
    hp'_pi : IsPiSystem p'
    h_gen' : Eq m' (MeasurableSpace.generateFrom p')
    h_pi_system_indep : ProbabilityTheory.Kernel.IndepSets (Set.iUnion fun n => p  …
    ⊢ Eq (iSup fun i => m i) (MeasurableSpace.generateFrom (Set.iUnion fun n => p  …
  -/
  exact (generateFrom_iUnion_measurableSet _).symm
  /-
    🎉 no goals
  -/


theorem iIndepSet.indep_generateFrom_lt [Preorder ι] {s : ι → Set Ω}
    (hsm : ∀ n, MeasurableSet (s n)) (hs : iIndepSet s κ μ) (i : ι) :
    Indep (generateFrom {s i}) (generateFrom { t | ∃ j < i, s j = t }) κ μ := by
  convert iIndepSet.indep_generateFrom_of_disjoint hsm hs {i} { j | j < i }
    (Set.disjoint_singleton_left.mpr (lt_irrefl _)) using 1
  /-
    case h.e'_4
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    inst✝ : Preorder ι
    s : ι → Set Ω
    hsm : ∀ (n : ι), MeasurableSet (s n)
    hs : ProbabilityTheory.Kernel.iIndepSet s κ μ
    i : ι
    ⊢ Eq (MeasurableSpace.generateFrom (Singleton.singleton (s i))) (MeasurableSpa …
  -/
  simp only [Set.mem_singleton_iff, exists_prop, exists_eq_left, Set.setOf_eq_eq_singleton']
  /-
    🎉 no goals
  -/


theorem iIndepSet.indep_generateFrom_le [LinearOrder ι] {s : ι → Set Ω}
    (hsm : ∀ n, MeasurableSet (s n)) (hs : iIndepSet s κ μ) (i : ι) {k : ι} (hk : i < k) :
    Indep (generateFrom {s k}) (generateFrom { t | ∃ j ≤ i, s j = t }) κ μ := by
  convert iIndepSet.indep_generateFrom_of_disjoint hsm hs {k} { j | j ≤ i }
      (Set.disjoint_singleton_left.mpr hk.not_le) using 1
  /-
    case h.e'_4
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    inst✝ : LinearOrder ι
    s : ι → Set Ω
    hsm : ∀ (n : ι), MeasurableSet (s n)
    hs : ProbabilityTheory.Kernel.iIndepSet s κ μ
    i k : ι
    hk : LT.lt i k
    ⊢ Eq (MeasurableSpace.generateFrom (Singleton.singleton (s k))) (MeasurableSpa …
  -/
  simp only [Set.mem_singleton_iff, exists_prop, exists_eq_left, Set.setOf_eq_eq_singleton']
  /-
    🎉 no goals
  -/


theorem iIndepSet.indep_generateFrom_le_nat {s : ℕ → Set Ω}
    (hsm : ∀ n, MeasurableSet (s n)) (hs : iIndepSet s κ μ) (n : ℕ) :
    Indep (generateFrom {s (n + 1)}) (generateFrom { t | ∃ k ≤ n, s k = t }) κ μ :=
  iIndepSet.indep_generateFrom_le hsm hs _ n.lt_succ_self


theorem indep_iSup_of_monotone [SemilatticeSup ι] {Ω} {m : ι → MeasurableSpace Ω}
    {m' m0 : MeasurableSpace Ω} {κ : Kernel α Ω} {μ : Measure α} [IsZeroOrMarkovKernel κ]
    (h_indep : ∀ i, Indep (m i) m' κ μ) (h_le : ∀ i, m i ≤ m0) (h_le' : m' ≤ m0)
    (hm : Monotone m) :
    Indep (⨆ i, m i) m' κ μ :=
  indep_iSup_of_directed_le h_indep h_le h_le' (Monotone.directed_le hm)


theorem indep_iSup_of_antitone [SemilatticeInf ι] {Ω} {m : ι → MeasurableSpace Ω}
    {m' m0 : MeasurableSpace Ω} {κ : Kernel α Ω} {μ : Measure α} [IsZeroOrMarkovKernel κ]
    (h_indep : ∀ i, Indep (m i) m' κ μ) (h_le : ∀ i, m i ≤ m0) (h_le' : m' ≤ m0)
    (hm : Antitone m) :
    Indep (⨆ i, m i) m' κ μ :=
  indep_iSup_of_directed_le h_indep h_le h_le' hm.directed_le


theorem iIndepSets.piiUnionInter_of_not_mem {π : ι → Set (Set Ω)} {a : ι} {S : Finset ι}
    (hp_ind : iIndepSets π κ μ) (haS : a ∉ S) :
    IndepSets (piiUnionInter π S) (π a) κ μ := by
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    π : ι → Set (Set Ω)
    a : ι
    S : Finset ι
    hp_ind : ProbabilityTheory.Kernel.iIndepSets π κ μ
    haS : Not (Membership.mem S a)
    ⊢ ProbabilityTheory.Kernel.IndepSets (piiUnionInter π ↑S) (π a) κ μ
  -/
  rintro t1 t2 ⟨s, hs_mem, ft1, hft1_mem, ht1_eq⟩ ht2_mem_pia
  /-
    case intro.intro.intro.intro
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    π : ι → Set (Set Ω)
    a : ι
    S : Finset ι
    hp_ind : ProbabilityTheory.Kernel.iIndepSets π κ μ
    haS : Not (Membership.mem S a)
    t1 t2 : Set Ω
    s : Finset ι
    hs_mem : HasSubset.Subset ↑s ↑S
    ft1 : ι → Set Ω
    hft1_mem : ∀ (x : ι), Membership.mem s x → Membership.mem (π x) (ft1 x)
    ht1_eq : Eq t1 (Set.iInter fun x => Set.iInter fun h => ft1 x)
    ht2_mem_pia : Membership.mem (π a) t2
    ⊢ Filter.Eventually (fun a => Eq ((κ a) (Inter.inter t1 t2)) (HMul.hMul ((κ a) …
  -/
  rw [Finset.coe_subset] at hs_mem
  classical
  let f := fun n => ite (n = a) t2 (ite (n ∈ s) (ft1 n) Set.univ)
  have h_f_mem : ∀ n ∈ insert a s, f n ∈ π n := by
    intro n hn_mem_insert
    dsimp only [f]
    cases' Finset.mem_insert.mp hn_mem_insert with hn_mem hn_mem
    · simp [hn_mem, ht2_mem_pia]
    · have hn_ne_a : n ≠ a := by rintro rfl; exact haS (hs_mem hn_mem)
      simp [hn_ne_a, hn_mem, hft1_mem n hn_mem]
  have h_f_mem_pi : ∀ n ∈ s, f n ∈ π n := fun x hxS => h_f_mem x (by simp [hxS])
  have h_t1 : t1 = ⋂ n ∈ s, f n := by
    suffices h_forall : ∀ n ∈ s, f n = ft1 n by
      rw [ht1_eq]
      ext x
      simp_rw [Set.mem_iInter]
      conv => lhs; intro i hns; rw [← h_forall i hns]
    intro n hnS
    have hn_ne_a : n ≠ a := by rintro rfl; exact haS (hs_mem hnS)
    simp_rw [f, if_pos hnS, if_neg hn_ne_a]
  have h_μ_t1 : ∀ᵐ a' ∂μ, κ a' t1 = ∏ n ∈ s, κ a' (f n) := by
    filter_upwards [hp_ind s h_f_mem_pi] with a' ha'
    rw [h_t1, ← ha']
  have h_t2 : t2 = f a := by simp [f]
  have h_μ_inter : ∀ᵐ a' ∂μ, κ a' (t1 ∩ t2) = ∏ n ∈ insert a s, κ a' (f n) := by
    have h_t1_inter_t2 : t1 ∩ t2 = ⋂ n ∈ insert a s, f n := by
      rw [h_t1, h_t2, Finset.set_biInter_insert, Set.inter_comm]
    filter_upwards [hp_ind (insert a s) h_f_mem] with a' ha'
    rw [h_t1_inter_t2, ← ha']
  have has : a ∉ s := fun has_mem => haS (hs_mem has_mem)
  filter_upwards [h_μ_t1, h_μ_inter] with a' ha1 ha2
  rw [ha2, Finset.prod_insert has, h_t2, mul_comm, ha1]


/-- The measurable space structures generated by independent pi-systems are independent. -/
theorem iIndepSets.iIndep (m : ι → MeasurableSpace Ω)
    (h_le : ∀ i, m i ≤ _mΩ) (π : ι → Set (Set Ω)) (h_pi : ∀ n, IsPiSystem (π n))
    (h_generate : ∀ i, m i = generateFrom (π i)) (h_ind : iIndepSets π κ μ) :
    iIndep m κ μ := by
  classical
  rcases eq_or_ne μ 0 with rfl | hμ
  · simp
  obtain ⟨η, η_eq, hη⟩ : ∃ (η : Kernel α Ω), κ =ᵐ[μ] η ∧ IsMarkovKernel η :=
    exists_ae_eq_isMarkovKernel h_ind.ae_isProbabilityMeasure hμ
  apply iIndep.congr (Filter.EventuallyEq.symm η_eq)
  intro s f
  refine Finset.induction ?_ ?_ s
  · simp only [Finset.not_mem_empty, Set.mem_setOf_eq, IsEmpty.forall_iff, implies_true,
      Set.iInter_of_empty, Set.iInter_univ, measure_univ, Finset.prod_empty,
      Filter.eventually_true, forall_true_left]
  · intro a S ha_notin_S h_rec hf_m
    have hf_m_S : ∀ x ∈ S, MeasurableSet[m x] (f x) := fun x hx => hf_m x (by simp [hx])
    let p := piiUnionInter π S
    set m_p := generateFrom p with hS_eq_generate
    have h_indep : Indep m_p (m a) η μ := by
      have hp : IsPiSystem p := isPiSystem_piiUnionInter π h_pi S
      have h_le' : ∀ i, generateFrom (π i) ≤ _mΩ := fun i ↦ (h_generate i).symm.trans_le (h_le i)
      have hm_p : m_p ≤ _mΩ := generateFrom_piiUnionInter_le π h_le' S
      exact IndepSets.indep hm_p (h_le a) hp (h_pi a) hS_eq_generate (h_generate a)
        (iIndepSets.piiUnionInter_of_not_mem (h_ind.congr η_eq) ha_notin_S)
    have h := h_indep.symm (f a) (⋂ n ∈ S, f n) (hf_m a (Finset.mem_insert_self a S)) ?_
    · filter_upwards [h_rec hf_m_S, h] with a' ha' h'
      rwa [Finset.set_biInter_insert, Finset.prod_insert ha_notin_S, ← ha']
    · have h_le_p : ∀ i ∈ S, m i ≤ m_p := by
        intros n hn
        rw [hS_eq_generate, h_generate n]
        exact le_generateFrom_piiUnionInter (S : Set ι) hn
      have h_S_f : ∀ i ∈ S, MeasurableSet[m_p] (f i) :=
        fun i hi ↦ (h_le_p i hi) (f i) (hf_m_S i hi)
      exact S.measurableSet_biInter h_S_f


theorem iIndepSet_iff_iIndepSets_singleton {_mΩ : MeasurableSpace Ω} {κ : Kernel α Ω}
    {μ : Measure α} {f : ι → Set Ω} (hf : ∀ i, MeasurableSet (f i)) :
    iIndepSet f κ μ ↔ iIndepSets (fun i ↦ {f i}) κ μ :=
  ⟨iIndep.iIndepSets fun _ ↦ rfl,
                                                       /-
                                                         α : Type u_1
                                                         Ω : Type u_2
                                                         ι : Type u_3
                                                         _mα : MeasurableSpace α
                                                         _mΩ : MeasurableSpace Ω
                                                         κ : ProbabilityTheory.Kernel α Ω
                                                         μ : MeasureTheory.Measure α
                                                         f : ι → Set Ω
                                                         hf : ∀ (i : ι), MeasurableSet (f i)
                                                         i : ι
                                                         ⊢ ∀ (t : Set Ω), Membership.mem (Singleton.singleton (f i)) t → MeasurableSet t
                                                       -/
    iIndepSets.iIndep _ (fun i ↦ generateFrom_le <| by rintro t (rfl : t = _); exact hf _) _
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
      (fun _ ↦ IsPiSystem.singleton _) fun _ ↦ rfl⟩


theorem iIndepSet.meas_biInter {_mΩ : MeasurableSpace Ω} {κ : Kernel α Ω}
    {μ : Measure α} {f : ι → Set Ω} (h : iIndepSet f κ μ) (s : Finset ι) :
    ∀ᵐ a ∂μ, κ a (⋂ i ∈ s, f i) = ∏ i ∈ s, κ a (f i) :=
                                          /-
                                            α : Type u_1
                                            Ω : Type u_2
                                            ι : Type u_3
                                            _mα : MeasurableSpace α
                                            _mΩ : MeasurableSpace Ω
                                            κ : ProbabilityTheory.Kernel α Ω
                                            μ : MeasureTheory.Measure α
                                            f : ι → Set Ω
                                            h : ProbabilityTheory.Kernel.iIndepSet f κ μ
                                            s : Finset ι
                                            ⊢ ∀ (i : ι), Membership.mem s i → Membership.mem (Singleton.singleton (f i)) ( …
                                          -/
  iIndep.iIndepSets (fun _ ↦ rfl) h _ (by simp)
                                          /-
                                            🎉 no goals
                                          -/


theorem iIndepSet_iff_meas_biInter {_mΩ : MeasurableSpace Ω} {κ : Kernel α Ω}
    {μ : Measure α} {f : ι → Set Ω} (hf : ∀ i, MeasurableSet (f i)) :
    iIndepSet f κ μ ↔ ∀ s, ∀ᵐ a ∂μ, κ a (⋂ i ∈ s, f i) = ∏ i ∈ s, κ a (f i) :=
  (iIndepSet_iff_iIndepSets_singleton hf).trans iIndepSets_singleton_iff


theorem iIndepSets.iIndepSet_of_mem {_mΩ : MeasurableSpace Ω} {κ : Kernel α Ω}
    {μ : Measure α} {π : ι → Set (Set Ω)} {f : ι → Set Ω}
    (hfπ : ∀ i, f i ∈ π i) (hf : ∀ i, MeasurableSet (f i)) (hπ : iIndepSets π κ μ) :
    iIndepSet f κ μ :=
  (iIndepSet_iff_meas_biInter hf).2 fun _t ↦ hπ.meas_biInter _ fun _i _ ↦ hfπ _


theorem indepSet_iff_indepSets_singleton {m0 : MeasurableSpace Ω} (hs_meas : MeasurableSet s)
    (ht_meas : MeasurableSet t) (κ : Kernel α Ω) (μ : Measure α)
    [IsZeroOrMarkovKernel κ] :
    IndepSet s t κ μ ↔ IndepSets {s} {t} κ μ :=
  ⟨Indep.indepSets, fun h =>
    IndepSets.indep
                                      /-
                                        α : Type u_1
                                        Ω : Type u_2
                                        _mα : MeasurableSpace α
                                        s t : Set Ω
                                        m0 : MeasurableSpace Ω
                                        hs_meas : MeasurableSet s
                                        ht_meas : MeasurableSet t
                                        κ : ProbabilityTheory.Kernel α Ω
                                        μ : MeasureTheory.Measure α
                                        inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel κ
                                        h : ProbabilityTheory.Kernel.IndepSets (Singleton.singleton s) (Singleton.sing …
                                        u : Set Ω
                                        hu : Membership.mem (Singleton.singleton s) u
                                        ⊢ MeasurableSet u
                                      -/
      (generateFrom_le fun u hu => by rwa [Set.mem_singleton_iff.mp hu])
                                      /-
                                        🎉 no goals
                                      -/
                                      /-
                                        α : Type u_1
                                        Ω : Type u_2
                                        _mα : MeasurableSpace α
                                        s t : Set Ω
                                        m0 : MeasurableSpace Ω
                                        hs_meas : MeasurableSet s
                                        ht_meas : MeasurableSet t
                                        κ : ProbabilityTheory.Kernel α Ω
                                        μ : MeasureTheory.Measure α
                                        inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel κ
                                        h : ProbabilityTheory.Kernel.IndepSets (Singleton.singleton s) (Singleton.sing …
                                        u : Set Ω
                                        hu : Membership.mem (Singleton.singleton t) u
                                        ⊢ MeasurableSet u
                                      -/
      (generateFrom_le fun u hu => by rwa [Set.mem_singleton_iff.mp hu])
                                      /-
                                        🎉 no goals
                                      -/
      (IsPiSystem.singleton s) (IsPiSystem.singleton t) rfl rfl h⟩


theorem indepSet_iff_measure_inter_eq_mul {_m0 : MeasurableSpace Ω} (hs_meas : MeasurableSet s)
    (ht_meas : MeasurableSet t) (κ : Kernel α Ω) (μ : Measure α)
    [IsZeroOrMarkovKernel κ] :
    IndepSet s t κ μ ↔ ∀ᵐ a ∂μ, κ a (s ∩ t) = κ a s * κ a t :=
  (indepSet_iff_indepSets_singleton hs_meas ht_meas κ μ).trans indepSets_singleton_iff


theorem IndepSet.measure_inter_eq_mul {_m0 : MeasurableSpace Ω} (κ : Kernel α Ω) (μ : Measure α)
    (h : IndepSet s t κ μ) : ∀ᵐ a ∂μ, κ a (s ∩ t) = κ a s * κ a t :=
                            /-
                              α : Type u_1
                              Ω : Type u_2
                              _mα : MeasurableSpace α
                              s t : Set Ω
                              _m0 : MeasurableSpace Ω
                              κ : ProbabilityTheory.Kernel α Ω
                              μ : MeasureTheory.Measure α
                              h : ProbabilityTheory.Kernel.IndepSet s t κ μ
                              ⊢ Membership.mem (Singleton.singleton s) s
                            -/
                            /-
                              🎉 no goals
                            -/
  Indep.indepSets h _ _ (by simp) (by simp)
                                      /-
                                        🎉 no goals
                                      -/


theorem IndepSets.indepSet_of_mem {_m0 : MeasurableSpace Ω} (hs : s ∈ S) (ht : t ∈ T)
    (hs_meas : MeasurableSet s) (ht_meas : MeasurableSet t)
    (κ : Kernel α Ω) (μ : Measure α) [IsZeroOrMarkovKernel κ]
    (h_indep : IndepSets S T κ μ) :
    IndepSet s t κ μ :=
  (indepSet_iff_measure_inter_eq_mul hs_meas ht_meas κ μ).mpr (h_indep s t hs ht)


theorem Indep.indepSet_of_measurableSet {m₁ m₂ _ : MeasurableSpace Ω} {κ : Kernel α Ω}
    {μ : Measure α}
    (h_indep : Indep m₁ m₂ κ μ) {s t : Set Ω} (hs : MeasurableSet[m₁] s)
    (ht : MeasurableSet[m₂] t) :
    IndepSet s t κ μ := by
  /-
    α : Type u_1
    Ω : Type u_2
    _mα : MeasurableSpace α
    m₁ m₂ x✝ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    h_indep : ProbabilityTheory.Kernel.Indep m₁ m₂ κ μ
    s t : Set Ω
    hs : MeasurableSet s
    ht : MeasurableSet t
    ⊢ ProbabilityTheory.Kernel.IndepSet s t κ μ
  -/
  refine fun s' t' hs' ht' => h_indep s' t' ?_ ?_
  · induction s', hs' using generateFrom_induction with
    | hC t ht => exact ht ▸ hs
    | empty => exact @MeasurableSet.empty _ m₁
    | compl u _ hu => exact hu.compl
    | iUnion f _ hf => exact .iUnion hf
  · induction t', ht' using generateFrom_induction with
    | hC s hs => exact hs ▸ ht
    | empty => exact @MeasurableSet.empty _ m₂
    | compl u _ hu => exact hu.compl
    | iUnion f _ hf => exact .iUnion hf


theorem indep_iff_forall_indepSet (m₁ m₂ : MeasurableSpace Ω) {_m0 : MeasurableSpace Ω}
    (κ : Kernel α Ω) (μ : Measure α) :
    Indep m₁ m₂ κ μ ↔ ∀ s t, MeasurableSet[m₁] s → MeasurableSet[m₂] t → IndepSet s t κ μ :=
  ⟨fun h => fun _s _t hs ht => h.indepSet_of_measurableSet hs ht, fun h s t hs ht =>
    h s t hs ht s t (measurableSet_generateFrom (Set.mem_singleton s))
      (measurableSet_generateFrom (Set.mem_singleton t))⟩


theorem indepFun_iff_measure_inter_preimage_eq_mul {mβ : MeasurableSpace β}
    {mβ' : MeasurableSpace β'} :
    IndepFun f g κ μ ↔
      ∀ s t, MeasurableSet s → MeasurableSet t
        → ∀ᵐ a ∂μ, κ a (f ⁻¹' s ∩ g ⁻¹' t) = κ a (f ⁻¹' s) * κ a (g ⁻¹' t) := by
  /-
    α : Type u_1
    Ω : Type u_2
    β : Type u_4
    β' : Type u_5
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    f : Ω → β
    g : Ω → β'
    mβ : MeasurableSpace β
    mβ' : MeasurableSpace β'
    ⊢ Iff (ProbabilityTheory.Kernel.IndepFun f g κ μ) (∀ (s : Set β) (t : Set β'), …
  -/
  constructor <;> intro h
    /-
      case mp
      α : Type u_1
      Ω : Type u_2
      β : Type u_4
      β' : Type u_5
      _mα : MeasurableSpace α
      _mΩ : MeasurableSpace Ω
      κ : ProbabilityTheory.Kernel α Ω
      μ : MeasureTheory.Measure α
      f : Ω → β
      g : Ω → β'
      mβ : MeasurableSpace β
      mβ' : MeasurableSpace β'
      h : ProbabilityTheory.Kernel.IndepFun f g κ μ
      ⊢ ∀ (s : Set β) (t : Set β'), MeasurableSet s → MeasurableSet t → Filter.Event …
    -/
  · refine fun s t hs ht => h (f ⁻¹' s) (g ⁻¹' t) ⟨s, hs, rfl⟩ ⟨t, ht, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      Ω : Type u_2
      β : Type u_4
      β' : Type u_5
      _mα : MeasurableSpace α
      _mΩ : MeasurableSpace Ω
      κ : ProbabilityTheory.Kernel α Ω
      μ : MeasureTheory.Measure α
      f : Ω → β
      g : Ω → β'
      mβ : MeasurableSpace β
      mβ' : MeasurableSpace β'
      h : ∀ (s : Set β) (t : Set β'), MeasurableSet s → MeasurableSet t → Filter.Eve …
      ⊢ ProbabilityTheory.Kernel.IndepFun f g κ μ
    -/
  · rintro _ _ ⟨s, hs, rfl⟩ ⟨t, ht, rfl⟩; exact h s t hs ht
                                          /-
                                            🎉 no goals
                                          -/


alias ⟨IndepFun.measure_inter_preimage_eq_mul, _⟩ := indepFun_iff_measure_inter_preimage_eq_mul


theorem iIndepFun_iff_measure_inter_preimage_eq_mul {ι : Type*} {β : ι → Type*}
    (m : ∀ x, MeasurableSpace (β x)) (f : ∀ i, Ω → β i) :
    iIndepFun m f κ μ ↔
      ∀ (S : Finset ι) {sets : ∀ i : ι, Set (β i)} (_H : ∀ i, i ∈ S → MeasurableSet[m i] (sets i)),
        ∀ᵐ a ∂μ, κ a (⋂ i ∈ S, (f i) ⁻¹' (sets i)) = ∏ i ∈ S, κ a ((f i) ⁻¹' (sets i)) := by
  /-
    α : Type u_1
    Ω : Type u_2
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    ι : Type u_8
    β : ι → Type u_9
    m : (x : ι) → MeasurableSpace (β x)
    f : (i : ι) → Ω → β i
    ⊢ Iff (ProbabilityTheory.Kernel.iIndepFun m f κ μ) (∀ (S : Finset ι) {sets : ( …
  -/
  refine ⟨fun h S sets h_meas => h _ fun i hi_mem => ⟨sets i, h_meas i hi_mem, rfl⟩, ?_⟩
  /-
    α : Type u_1
    Ω : Type u_2
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    ι : Type u_8
    β : ι → Type u_9
    m : (x : ι) → MeasurableSpace (β x)
    f : (i : ι) → Ω → β i
    ⊢ (∀ (S : Finset ι) {sets : (i : ι) → Set (β i)}, (∀ (i : ι), Membership.mem S …
  -/
  intro h S setsΩ h_meas
  classical
  let setsβ : ∀ i : ι, Set (β i) := fun i =>
    dite (i ∈ S) (fun hi_mem => (h_meas i hi_mem).choose) fun _ => Set.univ
  have h_measβ : ∀ i ∈ S, MeasurableSet[m i] (setsβ i) := by
    intro i hi_mem
    simp_rw [setsβ, dif_pos hi_mem]
    exact (h_meas i hi_mem).choose_spec.1
  have h_preim : ∀ i ∈ S, setsΩ i = f i ⁻¹' setsβ i := by
    intro i hi_mem
    simp_rw [setsβ, dif_pos hi_mem]
    exact (h_meas i hi_mem).choose_spec.2.symm
  have h_left_eq : ∀ a, κ a (⋂ i ∈ S, setsΩ i) = κ a (⋂ i ∈ S, (f i) ⁻¹' (setsβ i)) := by
    intro a
    congr with x
    simp_rw [Set.mem_iInter]
    constructor <;> intro h i hi_mem <;> specialize h i hi_mem
    · rwa [h_preim i hi_mem] at h
    · rwa [h_preim i hi_mem]
  have h_right_eq : ∀ a, (∏ i ∈ S, κ a (setsΩ i)) = ∏ i ∈ S, κ a ((f i) ⁻¹' (setsβ i)) := by
    refine fun a ↦ Finset.prod_congr rfl fun i hi_mem => ?_
    rw [h_preim i hi_mem]
  filter_upwards [h S h_measβ] with a ha
  rw [h_left_eq a, h_right_eq a, ha]


alias ⟨iIndepFun.measure_inter_preimage_eq_mul, _⟩ := iIndepFun_iff_measure_inter_preimage_eq_mul


lemma iIndepFun.comp {β γ : ι → Type*} {mβ : ∀ i, MeasurableSpace (β i)}
    {mγ : ∀ i, MeasurableSpace (γ i)} {f : ∀ i, Ω → β i}
    (h : iIndepFun mβ f κ μ) (g : ∀ i, β i → γ i) (hg : ∀ i, Measurable (g i)) :
    iIndepFun mγ (fun i ↦ g i ∘ f i) κ μ := by
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    β : ι → Type u_8
    γ : ι → Type u_9
    mβ : (i : ι) → MeasurableSpace (β i)
    mγ : (i : ι) → MeasurableSpace (γ i)
    f : (i : ι) → Ω → β i
    h : ProbabilityTheory.Kernel.iIndepFun mβ f κ μ
    g : (i : ι) → β i → γ i
    hg : ∀ (i : ι), Measurable (g i)
    ⊢ ProbabilityTheory.Kernel.iIndepFun mγ (fun i => Function.comp (g i) (f i)) κ μ
  -/
  rw [iIndepFun_iff_measure_inter_preimage_eq_mul] at h ⊢
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    β : ι → Type u_8
    γ : ι → Type u_9
    mβ : (i : ι) → MeasurableSpace (β i)
    mγ : (i : ι) → MeasurableSpace (γ i)
    f : (i : ι) → Ω → β i
    h : ∀ (S : Finset ι) {sets : (i : ι) → Set (β i)}, (∀ (i : ι), Membership.mem  …
    g : (i : ι) → β i → γ i
    hg : ∀ (i : ι), Measurable (g i)
    ⊢ ∀ (S : Finset ι) {sets : (i : ι) → Set (γ i)}, (∀ (i : ι), Membership.mem S  …
  -/
  refine fun t s hs ↦ ?_
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    β : ι → Type u_8
    γ : ι → Type u_9
    mβ : (i : ι) → MeasurableSpace (β i)
    mγ : (i : ι) → MeasurableSpace (γ i)
    f : (i : ι) → Ω → β i
    h : ∀ (S : Finset ι) {sets : (i : ι) → Set (β i)}, (∀ (i : ι), Membership.mem  …
    g : (i : ι) → β i → γ i
    hg : ∀ (i : ι), Measurable (g i)
    t : Finset ι
    s : (i : ι) → Set (γ i)
    hs : ∀ (i : ι), Membership.mem t i → MeasurableSet (s i)
    ⊢ Filter.Eventually (fun a => Eq ((κ a) (Set.iInter fun i => Set.iInter fun h  …
  -/
  have := h t (sets := fun i ↦ g i ⁻¹' (s i)) (fun i a ↦ hg i (hs i a))
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    β : ι → Type u_8
    γ : ι → Type u_9
    mβ : (i : ι) → MeasurableSpace (β i)
    mγ : (i : ι) → MeasurableSpace (γ i)
    f : (i : ι) → Ω → β i
    h : ∀ (S : Finset ι) {sets : (i : ι) → Set (β i)}, (∀ (i : ι), Membership.mem  …
    g : (i : ι) → β i → γ i
    hg : ∀ (i : ι), Measurable (g i)
    t : Finset ι
    s : (i : ι) → Set (γ i)
    hs : ∀ (i : ι), Membership.mem t i → MeasurableSet (s i)
    this : Filter.Eventually (fun a => Eq ((κ a) (Set.iInter fun i => Set.iInter f …
    ⊢ Filter.Eventually (fun a => Eq ((κ a) (Set.iInter fun i => Set.iInter fun h  …
  -/
  filter_upwards [this] with a ha
  /-
    case h
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    β : ι → Type u_8
    γ : ι → Type u_9
    mβ : (i : ι) → MeasurableSpace (β i)
    mγ : (i : ι) → MeasurableSpace (γ i)
    f : (i : ι) → Ω → β i
    h : ∀ (S : Finset ι) {sets : (i : ι) → Set (β i)}, (∀ (i : ι), Membership.mem  …
    g : (i : ι) → β i → γ i
    hg : ∀ (i : ι), Measurable (g i)
    t : Finset ι
    s : (i : ι) → Set (γ i)
    hs : ∀ (i : ι), Membership.mem t i → MeasurableSet (s i)
    this : Filter.Eventually (fun a => Eq ((κ a) (Set.iInter fun i => Set.iInter f …
    a : α
    ha : Eq ((κ a) (Set.iInter fun i => Set.iInter fun h => Set.preimage (f i) (Se …
    ⊢ Eq ((κ a) (Set.iInter fun i => Set.iInter fun h => Set.preimage (Function.co …
  -/
  simp_rw [Set.preimage_comp]
  /-
    case h
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    β : ι → Type u_8
    γ : ι → Type u_9
    mβ : (i : ι) → MeasurableSpace (β i)
    mγ : (i : ι) → MeasurableSpace (γ i)
    f : (i : ι) → Ω → β i
    h : ∀ (S : Finset ι) {sets : (i : ι) → Set (β i)}, (∀ (i : ι), Membership.mem  …
    g : (i : ι) → β i → γ i
    hg : ∀ (i : ι), Measurable (g i)
    t : Finset ι
    s : (i : ι) → Set (γ i)
    hs : ∀ (i : ι), Membership.mem t i → MeasurableSet (s i)
    this : Filter.Eventually (fun a => Eq ((κ a) (Set.iInter fun i => Set.iInter f …
    a : α
    ha : Eq ((κ a) (Set.iInter fun i => Set.iInter fun h => Set.preimage (f i) (Se …
    ⊢ Eq ((κ a) (Set.iInter fun i => Set.iInter fun x => Set.preimage (f i) (Set.p …
  -/
  exact ha
  /-
    🎉 no goals
  -/


theorem indepFun_iff_indepSet_preimage {mβ : MeasurableSpace β} {mβ' : MeasurableSpace β'}
    [IsZeroOrMarkovKernel κ] (hf : Measurable f) (hg : Measurable g) :
    IndepFun f g κ μ ↔
      ∀ s t, MeasurableSet s → MeasurableSet t → IndepSet (f ⁻¹' s) (g ⁻¹' t) κ μ := by
  /-
    α : Type u_1
    Ω : Type u_2
    β : Type u_4
    β' : Type u_5
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    f : Ω → β
    g : Ω → β'
    mβ : MeasurableSpace β
    mβ' : MeasurableSpace β'
    inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel κ
    hf : Measurable f
    hg : Measurable g
    ⊢ Iff (ProbabilityTheory.Kernel.IndepFun f g κ μ) (∀ (s : Set β) (t : Set β'), …
  -/
  refine indepFun_iff_measure_inter_preimage_eq_mul.trans ?_
  /-
    α : Type u_1
    Ω : Type u_2
    β : Type u_4
    β' : Type u_5
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    f : Ω → β
    g : Ω → β'
    mβ : MeasurableSpace β
    mβ' : MeasurableSpace β'
    inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel κ
    hf : Measurable f
    hg : Measurable g
    ⊢ Iff (∀ (s : Set β) (t : Set β'), MeasurableSet s → MeasurableSet t → Filter. …
  -/
  constructor <;> intro h s t hs ht <;> specialize h s t hs ht
    /-
      case mp
      α : Type u_1
      Ω : Type u_2
      β : Type u_4
      β' : Type u_5
      _mα : MeasurableSpace α
      _mΩ : MeasurableSpace Ω
      κ : ProbabilityTheory.Kernel α Ω
      μ : MeasureTheory.Measure α
      f : Ω → β
      g : Ω → β'
      mβ : MeasurableSpace β
      mβ' : MeasurableSpace β'
      inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel κ
      hf : Measurable f
      hg : Measurable g
      s : Set β
      t : Set β'
      hs : MeasurableSet s
      ht : MeasurableSet t
      h : Filter.Eventually (fun a => Eq ((κ a) (Inter.inter (Set.preimage f s) (Set …
      ⊢ ProbabilityTheory.Kernel.IndepSet (Set.preimage f s) (Set.preimage g t) κ μ
    -/
  · rwa [indepSet_iff_measure_inter_eq_mul (hf hs) (hg ht) κ μ]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      Ω : Type u_2
      β : Type u_4
      β' : Type u_5
      _mα : MeasurableSpace α
      _mΩ : MeasurableSpace Ω
      κ : ProbabilityTheory.Kernel α Ω
      μ : MeasureTheory.Measure α
      f : Ω → β
      g : Ω → β'
      mβ : MeasurableSpace β
      mβ' : MeasurableSpace β'
      inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel κ
      hf : Measurable f
      hg : Measurable g
      s : Set β
      t : Set β'
      hs : MeasurableSet s
      ht : MeasurableSet t
      h : ProbabilityTheory.Kernel.IndepSet (Set.preimage f s) (Set.preimage g t) κ μ
      ⊢ Filter.Eventually (fun a => Eq ((κ a) (Inter.inter (Set.preimage f s) (Set.p …
    -/
  · rwa [← indepSet_iff_measure_inter_eq_mul (hf hs) (hg ht) κ μ]
    /-
      🎉 no goals
    -/


@[symm]
nonrec theorem IndepFun.symm {_ : MeasurableSpace β} {_ : MeasurableSpace β'}
    (hfg : IndepFun f g κ μ) : IndepFun g f κ μ := hfg.symm


theorem IndepFun.ae_eq {mβ : MeasurableSpace β} {mβ' : MeasurableSpace β'}
    {f' : Ω → β} {g' : Ω → β'} (hfg : IndepFun f g κ μ)
    (hf : ∀ᵐ a ∂μ, f =ᵐ[κ a] f') (hg : ∀ᵐ a ∂μ, g =ᵐ[κ a] g') :
    IndepFun f' g' κ μ := by
  /-
    α : Type u_1
    Ω : Type u_2
    β : Type u_4
    β' : Type u_5
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    f : Ω → β
    g : Ω → β'
    mβ : MeasurableSpace β
    mβ' : MeasurableSpace β'
    f' : Ω → β
    g' : Ω → β'
    hfg : ProbabilityTheory.Kernel.IndepFun f g κ μ
    hf : Filter.Eventually (fun a => (MeasureTheory.ae (κ a)).EventuallyEq f f') ( …
    hg : Filter.Eventually (fun a => (MeasureTheory.ae (κ a)).EventuallyEq g g') ( …
    ⊢ ProbabilityTheory.Kernel.IndepFun f' g' κ μ
  -/
  rintro _ _ ⟨A, hA, rfl⟩ ⟨B, hB, rfl⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    Ω : Type u_2
    β : Type u_4
    β' : Type u_5
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    f : Ω → β
    g : Ω → β'
    mβ : MeasurableSpace β
    mβ' : MeasurableSpace β'
    f' : Ω → β
    g' : Ω → β'
    hfg : ProbabilityTheory.Kernel.IndepFun f g κ μ
    hf : Filter.Eventually (fun a => (MeasureTheory.ae (κ a)).EventuallyEq f f') ( …
    hg : Filter.Eventually (fun a => (MeasureTheory.ae (κ a)).EventuallyEq g g') ( …
    A : Set β
    hA : MeasurableSet A
    B : Set β'
    hB : MeasurableSet B
    ⊢ Filter.Eventually (fun a => Eq ((κ a) (Inter.inter (Set.preimage f' A) (Set. …
  -/
  filter_upwards [hf, hg, hfg _ _ ⟨_, hA, rfl⟩ ⟨_, hB, rfl⟩] with a hf' hg' hfg'
  /-
    case h
    α : Type u_1
    Ω : Type u_2
    β : Type u_4
    β' : Type u_5
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    f : Ω → β
    g : Ω → β'
    mβ : MeasurableSpace β
    mβ' : MeasurableSpace β'
    f' : Ω → β
    g' : Ω → β'
    hfg : ProbabilityTheory.Kernel.IndepFun f g κ μ
    hf : Filter.Eventually (fun a => (MeasureTheory.ae (κ a)).EventuallyEq f f') ( …
    hg : Filter.Eventually (fun a => (MeasureTheory.ae (κ a)).EventuallyEq g g') ( …
    A : Set β
    hA : MeasurableSet A
    B : Set β'
    hB : MeasurableSet B
    a : α
    hf' : (MeasureTheory.ae (κ a)).EventuallyEq f f'
    hg' : (MeasureTheory.ae (κ a)).EventuallyEq g g'
    hfg' : Eq ((κ a) (Inter.inter (Set.preimage f A) (Set.preimage g B))) (HMul.hM …
    ⊢ Eq ((κ a) (Inter.inter (Set.preimage f' A) (Set.preimage g' B))) (HMul.hMul  …
  -/
  have h1 : f ⁻¹' A =ᵐ[κ a] f' ⁻¹' A := hf'.fun_comp A
  /-
    case h
    α : Type u_1
    Ω : Type u_2
    β : Type u_4
    β' : Type u_5
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    f : Ω → β
    g : Ω → β'
    mβ : MeasurableSpace β
    mβ' : MeasurableSpace β'
    f' : Ω → β
    g' : Ω → β'
    hfg : ProbabilityTheory.Kernel.IndepFun f g κ μ
    hf : Filter.Eventually (fun a => (MeasureTheory.ae (κ a)).EventuallyEq f f') ( …
    hg : Filter.Eventually (fun a => (MeasureTheory.ae (κ a)).EventuallyEq g g') ( …
    A : Set β
    hA : MeasurableSet A
    B : Set β'
    hB : MeasurableSet B
    a : α
    hf' : (MeasureTheory.ae (κ a)).EventuallyEq f f'
    hg' : (MeasureTheory.ae (κ a)).EventuallyEq g g'
    hfg' : Eq ((κ a) (Inter.inter (Set.preimage f A) (Set.preimage g B))) (HMul.hM …
    h1 : (MeasureTheory.ae (κ a)).EventuallyEq (Set.preimage f A) (Set.preimage f' …
    ⊢ Eq ((κ a) (Inter.inter (Set.preimage f' A) (Set.preimage g' B))) (HMul.hMul  …
  -/
  have h2 : g ⁻¹' B =ᵐ[κ a] g' ⁻¹' B := hg'.fun_comp B
  /-
    case h
    α : Type u_1
    Ω : Type u_2
    β : Type u_4
    β' : Type u_5
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    f : Ω → β
    g : Ω → β'
    mβ : MeasurableSpace β
    mβ' : MeasurableSpace β'
    f' : Ω → β
    g' : Ω → β'
    hfg : ProbabilityTheory.Kernel.IndepFun f g κ μ
    hf : Filter.Eventually (fun a => (MeasureTheory.ae (κ a)).EventuallyEq f f') ( …
    hg : Filter.Eventually (fun a => (MeasureTheory.ae (κ a)).EventuallyEq g g') ( …
    A : Set β
    hA : MeasurableSet A
    B : Set β'
    hB : MeasurableSet B
    a : α
    hf' : (MeasureTheory.ae (κ a)).EventuallyEq f f'
    hg' : (MeasureTheory.ae (κ a)).EventuallyEq g g'
    hfg' : Eq ((κ a) (Inter.inter (Set.preimage f A) (Set.preimage g B))) (HMul.hM …
    h1 : (MeasureTheory.ae (κ a)).EventuallyEq (Set.preimage f A) (Set.preimage f' …
    h2 : (MeasureTheory.ae (κ a)).EventuallyEq (Set.preimage g B) (Set.preimage g' …
    ⊢ Eq ((κ a) (Inter.inter (Set.preimage f' A) (Set.preimage g' B))) (HMul.hMul  …
  -/
  rwa [← measure_congr h1, ← measure_congr h2, ← measure_congr (h1.inter h2)]
  /-
    🎉 no goals
  -/


theorem IndepFun.comp {mβ : MeasurableSpace β} {mβ' : MeasurableSpace β'}
    {mγ : MeasurableSpace γ} {mγ' : MeasurableSpace γ'} {φ : β → γ} {ψ : β' → γ'}
    (hfg : IndepFun f g κ μ) (hφ : Measurable φ) (hψ : Measurable ψ) :
    IndepFun (φ ∘ f) (ψ ∘ g) κ μ := by
  /-
    α : Type u_1
    Ω : Type u_2
    β : Type u_4
    β' : Type u_5
    γ : Type u_6
    γ' : Type u_7
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    f : Ω → β
    g : Ω → β'
    mβ : MeasurableSpace β
    mβ' : MeasurableSpace β'
    mγ : MeasurableSpace γ
    mγ' : MeasurableSpace γ'
    φ : β → γ
    ψ : β' → γ'
    hfg : ProbabilityTheory.Kernel.IndepFun f g κ μ
    hφ : Measurable φ
    hψ : Measurable ψ
    ⊢ ProbabilityTheory.Kernel.IndepFun (Function.comp φ f) (Function.comp ψ g) κ μ
  -/
  rintro _ _ ⟨A, hA, rfl⟩ ⟨B, hB, rfl⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    Ω : Type u_2
    β : Type u_4
    β' : Type u_5
    γ : Type u_6
    γ' : Type u_7
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    f : Ω → β
    g : Ω → β'
    mβ : MeasurableSpace β
    mβ' : MeasurableSpace β'
    mγ : MeasurableSpace γ
    mγ' : MeasurableSpace γ'
    φ : β → γ
    ψ : β' → γ'
    hfg : ProbabilityTheory.Kernel.IndepFun f g κ μ
    hφ : Measurable φ
    hψ : Measurable ψ
    A : Set γ
    hA : MeasurableSet A
    B : Set γ'
    hB : MeasurableSet B
    ⊢ Filter.Eventually (fun a => Eq ((κ a) (Inter.inter (Set.preimage (Function.c …
  -/
  apply hfg
    /-
      case intro.intro.intro.intro.a
      α : Type u_1
      Ω : Type u_2
      β : Type u_4
      β' : Type u_5
      γ : Type u_6
      γ' : Type u_7
      _mα : MeasurableSpace α
      _mΩ : MeasurableSpace Ω
      κ : ProbabilityTheory.Kernel α Ω
      μ : MeasureTheory.Measure α
      f : Ω → β
      g : Ω → β'
      mβ : MeasurableSpace β
      mβ' : MeasurableSpace β'
      mγ : MeasurableSpace γ
      mγ' : MeasurableSpace γ'
      φ : β → γ
      ψ : β' → γ'
      hfg : ProbabilityTheory.Kernel.IndepFun f g κ μ
      hφ : Measurable φ
      hψ : Measurable ψ
      A : Set γ
      hA : MeasurableSet A
      B : Set γ'
      hB : MeasurableSet B
      ⊢ Membership.mem (setOf fun s => MeasurableSet s) (Set.preimage (Function.comp …
    -/
  · exact ⟨φ ⁻¹' A, hφ hA, Set.preimage_comp.symm⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.a
      α : Type u_1
      Ω : Type u_2
      β : Type u_4
      β' : Type u_5
      γ : Type u_6
      γ' : Type u_7
      _mα : MeasurableSpace α
      _mΩ : MeasurableSpace Ω
      κ : ProbabilityTheory.Kernel α Ω
      μ : MeasureTheory.Measure α
      f : Ω → β
      g : Ω → β'
      mβ : MeasurableSpace β
      mβ' : MeasurableSpace β'
      mγ : MeasurableSpace γ
      mγ' : MeasurableSpace γ'
      φ : β → γ
      ψ : β' → γ'
      hfg : ProbabilityTheory.Kernel.IndepFun f g κ μ
      hφ : Measurable φ
      hψ : Measurable ψ
      A : Set γ
      hA : MeasurableSet A
      B : Set γ'
      hB : MeasurableSet B
      ⊢ Membership.mem (setOf fun s => MeasurableSet s) (Set.preimage (Function.comp …
    -/
  · exact ⟨ψ ⁻¹' B, hψ hB, Set.preimage_comp.symm⟩
    /-
      🎉 no goals
    -/


theorem IndepFun.neg_right {_mβ : MeasurableSpace β} {_mβ' : MeasurableSpace β'} [Neg β']
    [MeasurableNeg β'] (hfg : IndepFun f g κ μ)  :
    IndepFun f (-g) κ μ := hfg.comp measurable_id measurable_neg


theorem IndepFun.neg_left {_mβ : MeasurableSpace β} {_mβ' : MeasurableSpace β'} [Neg β]
    [MeasurableNeg β] (hfg : IndepFun f g κ μ) :
    IndepFun (-f) g κ μ := hfg.comp measurable_neg measurable_id


@[nontriviality]
lemma iIndepFun.of_subsingleton [IsMarkovKernel κ] [Subsingleton ι] : iIndepFun m f κ μ := by
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    β : ι → Type u_8
    m : (i : ι) → MeasurableSpace (β i)
    f : (i : ι) → Ω → β i
    inst✝¹ : ProbabilityTheory.IsMarkovKernel κ
    inst✝ : Subsingleton ι
    ⊢ ProbabilityTheory.Kernel.iIndepFun m f κ μ
  -/
  refine (iIndepFun_iff_measure_inter_preimage_eq_mul ..).2 fun s f' hf' ↦ ?_
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    β : ι → Type u_8
    m : (i : ι) → MeasurableSpace (β i)
    f : (i : ι) → Ω → β i
    inst✝¹ : ProbabilityTheory.IsMarkovKernel κ
    inst✝ : Subsingleton ι
    s : Finset ι
    f' : (i : ι) → Set (β i)
    hf' : ∀ (i : ι), Membership.mem s i → MeasurableSet (f' i)
    ⊢ Filter.Eventually (fun a => Eq ((κ a) (Set.iInter fun i => Set.iInter fun h  …
  -/
  obtain rfl | ⟨x, hx⟩ := s.eq_empty_or_nonempty
    /-
      case inl
      α : Type u_1
      Ω : Type u_2
      ι : Type u_3
      _mα : MeasurableSpace α
      _mΩ : MeasurableSpace Ω
      κ : ProbabilityTheory.Kernel α Ω
      μ : MeasureTheory.Measure α
      β : ι → Type u_8
      m : (i : ι) → MeasurableSpace (β i)
      f : (i : ι) → Ω → β i
      inst✝¹ : ProbabilityTheory.IsMarkovKernel κ
      inst✝ : Subsingleton ι
      f' : (i : ι) → Set (β i)
      hf' : ∀ (i : ι), Membership.mem EmptyCollection.emptyCollection i → Measurable …
      ⊢ Filter.Eventually (fun a => Eq ((κ a) (Set.iInter fun i => Set.iInter fun h  …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr.intro
      α : Type u_1
      Ω : Type u_2
      ι : Type u_3
      _mα : MeasurableSpace α
      _mΩ : MeasurableSpace Ω
      κ : ProbabilityTheory.Kernel α Ω
      μ : MeasureTheory.Measure α
      β : ι → Type u_8
      m : (i : ι) → MeasurableSpace (β i)
      f : (i : ι) → Ω → β i
      inst✝¹ : ProbabilityTheory.IsMarkovKernel κ
      inst✝ : Subsingleton ι
      s : Finset ι
      f' : (i : ι) → Set (β i)
      hf' : ∀ (i : ι), Membership.mem s i → MeasurableSet (f' i)
      x : ι
      hx : Membership.mem s x
      ⊢ Filter.Eventually (fun a => Eq ((κ a) (Set.iInter fun i => Set.iInter fun h  …
    -/
  · have : s = {x} := by ext y; simp [Subsingleton.elim y x, hx]
    /-
      case inr.intro
      α : Type u_1
      Ω : Type u_2
      ι : Type u_3
      _mα : MeasurableSpace α
      _mΩ : MeasurableSpace Ω
      κ : ProbabilityTheory.Kernel α Ω
      μ : MeasureTheory.Measure α
      β : ι → Type u_8
      m : (i : ι) → MeasurableSpace (β i)
      f : (i : ι) → Ω → β i
      inst✝¹ : ProbabilityTheory.IsMarkovKernel κ
      inst✝ : Subsingleton ι
      s : Finset ι
      f' : (i : ι) → Set (β i)
      hf' : ∀ (i : ι), Membership.mem s i → MeasurableSet (f' i)
      x : ι
      hx : Membership.mem s x
      this : Eq s (Singleton.singleton x)
      ⊢ Filter.Eventually (fun a => Eq ((κ a) (Set.iInter fun i => Set.iInter fun h  …
    -/
    simp [this]
    /-
      🎉 no goals
    -/


/-- If `f` is a family of mutually independent random variables (`iIndepFun m f μ`) and `S, T` are
two disjoint finite index sets, then the tuple formed by `f i` for `i ∈ S` is independent of the
tuple `(f i)_i` for `i ∈ T`. -/
theorem iIndepFun.indepFun_finset (S T : Finset ι) (hST : Disjoint S T)
    (hf_Indep : iIndepFun m f κ μ) (hf_meas : ∀ i, Measurable (f i)) :
    IndepFun (fun a (i : S) => f i a) (fun a (i : T) => f i a) κ μ := by
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    β : ι → Type u_8
    m : (i : ι) → MeasurableSpace (β i)
    f : (i : ι) → Ω → β i
    S T : Finset ι
    hST : Disjoint S T
    hf_Indep : ProbabilityTheory.Kernel.iIndepFun m f κ μ
    hf_meas : ∀ (i : ι), Measurable (f i)
    ⊢ ProbabilityTheory.Kernel.IndepFun (fun a i => f (↑i) a) (fun a i => f (↑i) a …
  -/
  rcases eq_or_ne μ 0 with rfl | hμ
    /-
      case inl
      α : Type u_1
      Ω : Type u_2
      ι : Type u_3
      _mα : MeasurableSpace α
      _mΩ : MeasurableSpace Ω
      κ : ProbabilityTheory.Kernel α Ω
      β : ι → Type u_8
      m : (i : ι) → MeasurableSpace (β i)
      f : (i : ι) → Ω → β i
      S T : Finset ι
      hST : Disjoint S T
      hf_meas : ∀ (i : ι), Measurable (f i)
      hf_Indep : ProbabilityTheory.Kernel.iIndepFun m f κ 0
      ⊢ ProbabilityTheory.Kernel.IndepFun (fun a i => f (↑i) a) (fun a i => f (↑i) a …
    -/
  · simp
    /-
      🎉 no goals
    -/
  obtain ⟨η, η_eq, hη⟩ : ∃ (η : Kernel α Ω), κ =ᵐ[μ] η ∧ IsMarkovKernel η :=
    exists_ae_eq_isMarkovKernel hf_Indep.ae_isProbabilityMeasure hμ
  /-
    case inr.intro.intro
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    β : ι → Type u_8
    m : (i : ι) → MeasurableSpace (β i)
    f : (i : ι) → Ω → β i
    S T : Finset ι
    hST : Disjoint S T
    hf_Indep : ProbabilityTheory.Kernel.iIndepFun m f κ μ
    hf_meas : ∀ (i : ι), Measurable (f i)
    hμ : Ne μ 0
    η : ProbabilityTheory.Kernel α Ω
    η_eq : (MeasureTheory.ae μ).EventuallyEq ⇑κ ⇑η
    hη : ProbabilityTheory.IsMarkovKernel η
    ⊢ ProbabilityTheory.Kernel.IndepFun (fun a i => f (↑i) a) (fun a i => f (↑i) a …
  -/
  apply IndepFun.congr (Filter.EventuallyEq.symm η_eq)
  -- We introduce π-systems, built from the π-system of boxes which generates `MeasurableSpace.pi`.
  let πSβ := Set.pi (Set.univ : Set S) ''
    Set.pi (Set.univ : Set S) fun i => { s : Set (β i) | MeasurableSet[m i] s }
  /-
    case inr.intro.intro
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    β : ι → Type u_8
    m : (i : ι) → MeasurableSpace (β i)
    f : (i : ι) → Ω → β i
    S T : Finset ι
    hST : Disjoint S T
    hf_Indep : ProbabilityTheory.Kernel.iIndepFun m f κ μ
    hf_meas : ∀ (i : ι), Measurable (f i)
    hμ : Ne μ 0
    η : ProbabilityTheory.Kernel α Ω
    η_eq : (MeasureTheory.ae μ).EventuallyEq ⇑κ ⇑η
    hη : ProbabilityTheory.IsMarkovKernel η
    πSβ : Set (Set ((i : Subtype fun x => Membership.mem S x) → β ↑i)) := Set.imag …
    ⊢ ProbabilityTheory.Kernel.IndepFun (fun a i => f (↑i) a) (fun a i => f (↑i) a …
  -/
  let πS := { s : Set Ω | ∃ t ∈ πSβ, (fun a (i : S) => f i a) ⁻¹' t = s }
  /-
    case inr.intro.intro
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    β : ι → Type u_8
    m : (i : ι) → MeasurableSpace (β i)
    f : (i : ι) → Ω → β i
    S T : Finset ι
    hST : Disjoint S T
    hf_Indep : ProbabilityTheory.Kernel.iIndepFun m f κ μ
    hf_meas : ∀ (i : ι), Measurable (f i)
    hμ : Ne μ 0
    η : ProbabilityTheory.Kernel α Ω
    η_eq : (MeasureTheory.ae μ).EventuallyEq ⇑κ ⇑η
    hη : ProbabilityTheory.IsMarkovKernel η
    πSβ : Set (Set ((i : Subtype fun x => Membership.mem S x) → β ↑i)) := Set.imag …
    πS : Set (Set Ω) := setOf fun s => Exists fun t => And (Membership.mem πSβ t)  …
    ⊢ ProbabilityTheory.Kernel.IndepFun (fun a i => f (↑i) a) (fun a i => f (↑i) a …
  -/
  have hπS_pi : IsPiSystem πS := by exact IsPiSystem.comap (@isPiSystem_pi _ _ ?_) _
  have hπS_gen : (MeasurableSpace.pi.comap fun a (i : S) => f i a) = generateFrom πS := by
    rw [generateFrom_pi.symm, comap_generateFrom]
    congr
  let πTβ := Set.pi (Set.univ : Set T) ''
      Set.pi (Set.univ : Set T) fun i => { s : Set (β i) | MeasurableSet[m i] s }
  /-
    case inr.intro.intro
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    β : ι → Type u_8
    m : (i : ι) → MeasurableSpace (β i)
    f : (i : ι) → Ω → β i
    S T : Finset ι
    hST : Disjoint S T
    hf_Indep : ProbabilityTheory.Kernel.iIndepFun m f κ μ
    hf_meas : ∀ (i : ι), Measurable (f i)
    hμ : Ne μ 0
    η : ProbabilityTheory.Kernel α Ω
    η_eq : (MeasureTheory.ae μ).EventuallyEq ⇑κ ⇑η
    hη : ProbabilityTheory.IsMarkovKernel η
    πSβ : Set (Set ((i : Subtype fun x => Membership.mem S x) → β ↑i)) := Set.imag …
    πS : Set (Set Ω) := setOf fun s => Exists fun t => And (Membership.mem πSβ t)  …
    hπS_pi : IsPiSystem πS
    hπS_gen : Eq (MeasurableSpace.comap (fun a i => f (↑i) a) MeasurableSpace.pi)  …
    πTβ : Set (Set ((i : Subtype fun x => Membership.mem T x) → β ↑i)) := Set.imag …
    ⊢ ProbabilityTheory.Kernel.IndepFun (fun a i => f (↑i) a) (fun a i => f (↑i) a …
  -/
  let πT := { s : Set Ω | ∃ t ∈ πTβ, (fun a (i : T) => f i a) ⁻¹' t = s }
  /-
    case inr.intro.intro
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    β : ι → Type u_8
    m : (i : ι) → MeasurableSpace (β i)
    f : (i : ι) → Ω → β i
    S T : Finset ι
    hST : Disjoint S T
    hf_Indep : ProbabilityTheory.Kernel.iIndepFun m f κ μ
    hf_meas : ∀ (i : ι), Measurable (f i)
    hμ : Ne μ 0
    η : ProbabilityTheory.Kernel α Ω
    η_eq : (MeasureTheory.ae μ).EventuallyEq ⇑κ ⇑η
    hη : ProbabilityTheory.IsMarkovKernel η
    πSβ : Set (Set ((i : Subtype fun x => Membership.mem S x) → β ↑i)) := Set.imag …
    πS : Set (Set Ω) := setOf fun s => Exists fun t => And (Membership.mem πSβ t)  …
    hπS_pi : IsPiSystem πS
    hπS_gen : Eq (MeasurableSpace.comap (fun a i => f (↑i) a) MeasurableSpace.pi)  …
    πTβ : Set (Set ((i : Subtype fun x => Membership.mem T x) → β ↑i)) := Set.imag …
    πT : Set (Set Ω) := setOf fun s => Exists fun t => And (Membership.mem πTβ t)  …
    ⊢ ProbabilityTheory.Kernel.IndepFun (fun a i => f (↑i) a) (fun a i => f (↑i) a …
  -/
  have hπT_pi : IsPiSystem πT := by exact IsPiSystem.comap (@isPiSystem_pi _ _ ?_) _
  have hπT_gen : (MeasurableSpace.pi.comap fun a (i : T) => f i a) = generateFrom πT := by
    rw [generateFrom_pi.symm, comap_generateFrom]
    congr
  -- To prove independence, we prove independence of the generating π-systems.
  refine IndepSets.indep (Measurable.comap_le (measurable_pi_iff.mpr fun i => hf_meas i))
    (Measurable.comap_le (measurable_pi_iff.mpr fun i => hf_meas i)) hπS_pi hπT_pi hπS_gen hπT_gen
    ?_
  /-
    case inr.intro.intro
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    β : ι → Type u_8
    m : (i : ι) → MeasurableSpace (β i)
    f : (i : ι) → Ω → β i
    S T : Finset ι
    hST : Disjoint S T
    hf_Indep : ProbabilityTheory.Kernel.iIndepFun m f κ μ
    hf_meas : ∀ (i : ι), Measurable (f i)
    hμ : Ne μ 0
    η : ProbabilityTheory.Kernel α Ω
    η_eq : (MeasureTheory.ae μ).EventuallyEq ⇑κ ⇑η
    hη : ProbabilityTheory.IsMarkovKernel η
    πSβ : Set (Set ((i : Subtype fun x => Membership.mem S x) → β ↑i)) := Set.imag …
    πS : Set (Set Ω) := setOf fun s => Exists fun t => And (Membership.mem πSβ t)  …
    hπS_pi : IsPiSystem πS
    hπS_gen : Eq (MeasurableSpace.comap (fun a i => f (↑i) a) MeasurableSpace.pi)  …
    πTβ : Set (Set ((i : Subtype fun x => Membership.mem T x) → β ↑i)) := Set.imag …
    πT : Set (Set Ω) := setOf fun s => Exists fun t => And (Membership.mem πTβ t)  …
    hπT_pi : IsPiSystem πT
    hπT_gen : Eq (MeasurableSpace.comap (fun a i => f (↑i) a) MeasurableSpace.pi)  …
    ⊢ ProbabilityTheory.Kernel.IndepSets πS πT η μ
  -/
  rintro _ _ ⟨s, ⟨sets_s, hs1, hs2⟩, rfl⟩ ⟨t, ⟨sets_t, ht1, ht2⟩, rfl⟩
  /-
    case inr.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    β : ι → Type u_8
    m : (i : ι) → MeasurableSpace (β i)
    f : (i : ι) → Ω → β i
    S T : Finset ι
    hST : Disjoint S T
    hf_Indep : ProbabilityTheory.Kernel.iIndepFun m f κ μ
    hf_meas : ∀ (i : ι), Measurable (f i)
    hμ : Ne μ 0
    η : ProbabilityTheory.Kernel α Ω
    η_eq : (MeasureTheory.ae μ).EventuallyEq ⇑κ ⇑η
    hη : ProbabilityTheory.IsMarkovKernel η
    πSβ : Set (Set ((i : Subtype fun x => Membership.mem S x) → β ↑i)) := Set.imag …
    πS : Set (Set Ω) := setOf fun s => Exists fun t => And (Membership.mem πSβ t)  …
    hπS_pi : IsPiSystem πS
    hπS_gen : Eq (MeasurableSpace.comap (fun a i => f (↑i) a) MeasurableSpace.pi)  …
    πTβ : Set (Set ((i : Subtype fun x => Membership.mem T x) → β ↑i)) := Set.imag …
    πT : Set (Set Ω) := setOf fun s => Exists fun t => And (Membership.mem πTβ t)  …
    hπT_pi : IsPiSystem πT
    hπT_gen : Eq (MeasurableSpace.comap (fun a i => f (↑i) a) MeasurableSpace.pi)  …
    s : Set ((i : Subtype fun x => Membership.mem S x) → β ↑i)
    sets_s : (i : Subtype fun x => Membership.mem S x) → Set (β ↑i)
    hs1 : Membership.mem (Set.univ.pi fun i => setOf fun s => MeasurableSet s) set …
    hs2 : Eq (Set.univ.pi sets_s) s
    t : Set ((i : Subtype fun x => Membership.mem T x) → β ↑i)
    sets_t : (i : Subtype fun x => Membership.mem T x) → Set (β ↑i)
    ht1 : Membership.mem (Set.univ.pi fun i => setOf fun s => MeasurableSet s) set …
    ht2 : Eq (Set.univ.pi sets_t) t
    ⊢ Filter.Eventually (fun a => Eq ((η a) (Inter.inter (Set.preimage (fun a i => …
  -/
  simp only [Set.mem_univ_pi, Set.mem_setOf_eq] at hs1 ht1
  /-
    case inr.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    β : ι → Type u_8
    m : (i : ι) → MeasurableSpace (β i)
    f : (i : ι) → Ω → β i
    S T : Finset ι
    hST : Disjoint S T
    hf_Indep : ProbabilityTheory.Kernel.iIndepFun m f κ μ
    hf_meas : ∀ (i : ι), Measurable (f i)
    hμ : Ne μ 0
    η : ProbabilityTheory.Kernel α Ω
    η_eq : (MeasureTheory.ae μ).EventuallyEq ⇑κ ⇑η
    hη : ProbabilityTheory.IsMarkovKernel η
    πSβ : Set (Set ((i : Subtype fun x => Membership.mem S x) → β ↑i)) := Set.imag …
    πS : Set (Set Ω) := setOf fun s => Exists fun t => And (Membership.mem πSβ t)  …
    hπS_pi : IsPiSystem πS
    hπS_gen : Eq (MeasurableSpace.comap (fun a i => f (↑i) a) MeasurableSpace.pi)  …
    πTβ : Set (Set ((i : Subtype fun x => Membership.mem T x) → β ↑i)) := Set.imag …
    πT : Set (Set Ω) := setOf fun s => Exists fun t => And (Membership.mem πTβ t)  …
    hπT_pi : IsPiSystem πT
    hπT_gen : Eq (MeasurableSpace.comap (fun a i => f (↑i) a) MeasurableSpace.pi)  …
    s : Set ((i : Subtype fun x => Membership.mem S x) → β ↑i)
    sets_s : (i : Subtype fun x => Membership.mem S x) → Set (β ↑i)
    hs2 : Eq (Set.univ.pi sets_s) s
    t : Set ((i : Subtype fun x => Membership.mem T x) → β ↑i)
    sets_t : (i : Subtype fun x => Membership.mem T x) → Set (β ↑i)
    ht2 : Eq (Set.univ.pi sets_t) t
    hs1 : ∀ (i : Subtype fun x => Membership.mem S x), MeasurableSet (sets_s i)
    ht1 : ∀ (i : Subtype fun x => Membership.mem T x), MeasurableSet (sets_t i)
    ⊢ Filter.Eventually (fun a => Eq ((η a) (Inter.inter (Set.preimage (fun a i => …
  -/
  rw [← hs2, ← ht2]
  classical
  let sets_s' : ∀ i : ι, Set (β i) := fun i =>
    dite (i ∈ S) (fun hi => sets_s ⟨i, hi⟩) fun _ => Set.univ
  have h_sets_s'_eq : ∀ {i} (hi : i ∈ S), sets_s' i = sets_s ⟨i, hi⟩ := by
    intro i hi; simp_rw [sets_s', dif_pos hi]
  have h_sets_s'_univ : ∀ {i} (_hi : i ∈ T), sets_s' i = Set.univ := by
    intro i hi; simp_rw [sets_s', dif_neg (Finset.disjoint_right.mp hST hi)]
  let sets_t' : ∀ i : ι, Set (β i) := fun i =>
    dite (i ∈ T) (fun hi => sets_t ⟨i, hi⟩) fun _ => Set.univ
  have h_sets_t'_univ : ∀ {i} (_hi : i ∈ S), sets_t' i = Set.univ := by
    intro i hi; simp_rw [sets_t', dif_neg (Finset.disjoint_left.mp hST hi)]
  have h_meas_s' : ∀ i ∈ S, MeasurableSet (sets_s' i) := by
    intro i hi; rw [h_sets_s'_eq hi]; exact hs1 _
  have h_meas_t' : ∀ i ∈ T, MeasurableSet (sets_t' i) := by
    intro i hi; simp_rw [sets_t', dif_pos hi]; exact ht1 _
  have h_eq_inter_S : (fun (ω : Ω) (i : ↥S) =>
    f (↑i) ω) ⁻¹' Set.pi Set.univ sets_s = ⋂ i ∈ S, f i ⁻¹' sets_s' i := by
    ext1 x
    simp_rw [Set.mem_preimage, Set.mem_univ_pi, Set.mem_iInter]
    constructor <;> intro h
    · intro i hi; simp only [h_sets_s'_eq hi, Set.mem_preimage]; exact h ⟨i, hi⟩
    · rintro ⟨i, hi⟩; specialize h i hi; simp only [sets_s'] at h; rwa [dif_pos hi] at h
  have h_eq_inter_T : (fun (ω : Ω) (i : ↥T) => f (↑i) ω) ⁻¹' Set.pi Set.univ sets_t
    = ⋂ i ∈ T, f i ⁻¹' sets_t' i := by
    ext1 x
    simp only [Set.mem_preimage, Set.mem_univ_pi, Set.mem_iInter]
    constructor <;> intro h
    · intro i hi; simp_rw [sets_t', dif_pos hi]; exact h ⟨i, hi⟩
    · rintro ⟨i, hi⟩; specialize h i hi; simp_rw [sets_t', dif_pos hi] at h; exact h
  replace hf_Indep := hf_Indep.congr η_eq
  rw [iIndepFun_iff_measure_inter_preimage_eq_mul] at hf_Indep
  have h_Inter_inter :
    ((⋂ i ∈ S, f i ⁻¹' sets_s' i) ∩ ⋂ i ∈ T, f i ⁻¹' sets_t' i) =
      ⋂ i ∈ S ∪ T, f i ⁻¹' (sets_s' i ∩ sets_t' i) := by
    ext1 x
    simp_rw [Set.mem_inter_iff, Set.mem_iInter, Set.mem_preimage, Finset.mem_union]
    constructor <;> intro h
    · intro i hi
      cases' hi with hiS hiT
      · replace h := h.1 i hiS
        simp_rw [sets_s', sets_t', dif_pos hiS, dif_neg (Finset.disjoint_left.mp hST hiS)]
        simp only [sets_s'] at h
        exact ⟨by rwa [dif_pos hiS] at h, Set.mem_univ _⟩
      · replace h := h.2 i hiT
        simp_rw [sets_s', sets_t', dif_pos hiT, dif_neg (Finset.disjoint_right.mp hST hiT)]
        simp only [sets_t'] at h
        exact ⟨Set.mem_univ _, by rwa [dif_pos hiT] at h⟩
    · exact ⟨fun i hi => (h i (Or.inl hi)).1, fun i hi => (h i (Or.inr hi)).2⟩
  have h_meas_inter : ∀ i ∈ S ∪ T, MeasurableSet (sets_s' i ∩ sets_t' i) := by
    intros i hi_mem
    rw [Finset.mem_union] at hi_mem
    cases' hi_mem with hi_mem hi_mem
    · rw [h_sets_t'_univ hi_mem, Set.inter_univ]
      exact h_meas_s' i hi_mem
    · rw [h_sets_s'_univ hi_mem, Set.univ_inter]
      exact h_meas_t' i hi_mem
  filter_upwards [hf_Indep S h_meas_s', hf_Indep T h_meas_t', hf_Indep (S ∪ T) h_meas_inter]
    with a h_indepS h_indepT h_indepST
  rw [h_eq_inter_S, h_eq_inter_T, h_indepS, h_indepT, h_Inter_inter, h_indepST,
    Finset.prod_union hST]
  congr 1
  · refine Finset.prod_congr rfl fun i hi => ?_
    rw [h_sets_t'_univ hi, Set.inter_univ]
  · refine Finset.prod_congr rfl fun i hi => ?_
    rw [h_sets_s'_univ hi, Set.univ_inter]


theorem iIndepFun.indepFun_prod_mk (hf_Indep : iIndepFun m f κ μ)
    (hf_meas : ∀ i, Measurable (f i)) (i j k : ι) (hik : i ≠ k) (hjk : j ≠ k) :
    IndepFun (fun a => (f i a, f j a)) (f k) κ μ := by
  classical
  have h_right : f k =
    (fun p : ∀ j : ({k} : Finset ι), β j => p ⟨k, Finset.mem_singleton_self k⟩) ∘
    fun a (j : ({k} : Finset ι)) => f j a := rfl
  have h_meas_right :  Measurable fun p : ∀ j : ({k} : Finset ι),
    β j => p ⟨k, Finset.mem_singleton_self k⟩ := measurable_pi_apply _
  let s : Finset ι := {i, j}
  have h_left : (fun ω => (f i ω, f j ω)) = (fun p : ∀ l : s, β l =>
    (p ⟨i, Finset.mem_insert_self i _⟩,
    p ⟨j, Finset.mem_insert_of_mem (Finset.mem_singleton_self _)⟩)) ∘ fun a (j : s) => f j a := by
    ext1 a
    simp only [Prod.mk.inj_iff]
    constructor
  have h_meas_left : Measurable fun p : ∀ l : s, β l =>
    (p ⟨i, Finset.mem_insert_self i _⟩,
    p ⟨j, Finset.mem_insert_of_mem (Finset.mem_singleton_self _)⟩) :=
      Measurable.prod (measurable_pi_apply _) (measurable_pi_apply _)
  rw [h_left, h_right]
  refine (hf_Indep.indepFun_finset s {k} ?_ hf_meas).comp h_meas_left h_meas_right
  rw [Finset.disjoint_singleton_right]
  simp only [s, Finset.mem_insert, Finset.mem_singleton, not_or]
  exact ⟨hik.symm, hjk.symm⟩


open Finset in
lemma iIndepFun.indepFun_prod_mk_prod_mk (hf_indep : iIndepFun m f κ μ)
    (hf_meas : ∀ i, Measurable (f i))
    (i j k l : ι) (hik : i ≠ k) (hil : i ≠ l) (hjk : j ≠ k) (hjl : j ≠ l) :
    IndepFun (fun a ↦ (f i a, f j a)) (fun a ↦ (f k a, f l a)) κ μ := by
  classical
  let g (i j : ι) (v : Π x : ({i, j} : Finset ι), β x) : β i × β j :=
    ⟨v ⟨i, mem_insert_self _ _⟩, v ⟨j, mem_insert_of_mem <| mem_singleton_self _⟩⟩
  have hg (i j : ι) : Measurable (g i j) := by fun_prop
  exact (hf_indep.indepFun_finset {i, j} {k, l} (by aesop) hf_meas).comp (hg i j) (hg k l)


@[to_additive]
lemma iIndepFun.indepFun_mul_left (hf_indep : iIndepFun (fun _ ↦ m) f κ μ)
    (hf_meas : ∀ i, Measurable (f i)) (i j k : ι) (hik : i ≠ k) (hjk : j ≠ k) :
    IndepFun (f i * f j) (f k) κ μ := by
  have : IndepFun (fun ω => (f i ω, f j ω)) (f k) κ μ :=
    hf_indep.indepFun_prod_mk hf_meas i j k hik hjk
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    β : Type u_8
    m : MeasurableSpace β
    inst✝¹ : Mul β
    inst✝ : MeasurableMul₂ β
    f : ι → Ω → β
    hf_indep : ProbabilityTheory.Kernel.iIndepFun (fun x => m) f κ μ
    hf_meas : ∀ (i : ι), Measurable (f i)
    i j k : ι
    hik : Ne i k
    hjk : Ne j k
    this : ProbabilityTheory.Kernel.IndepFun (fun ω => { fst := f i ω, snd := f j  …
    ⊢ ProbabilityTheory.Kernel.IndepFun (HMul.hMul (f i) (f j)) (f k) κ μ
  -/
  simpa using this.comp (measurable_fst.mul measurable_snd) measurable_id
  /-
    🎉 no goals
  -/


@[to_additive]
lemma iIndepFun.indepFun_mul_right (hf_indep : iIndepFun (fun _ ↦ m) f κ μ)
    (hf_meas : ∀ i, Measurable (f i)) (i j k : ι) (hij : i ≠ j) (hik : i ≠ k) :
    IndepFun (f i) (f j * f k) κ μ :=
  (hf_indep.indepFun_mul_left hf_meas _ _ _ hij.symm hik.symm).symm


@[to_additive]
lemma iIndepFun.indepFun_mul_mul (hf_indep : iIndepFun (fun _ ↦ m) f κ μ)
    (hf_meas : ∀ i, Measurable (f i))
    (i j k l : ι) (hik : i ≠ k) (hil : i ≠ l) (hjk : j ≠ k) (hjl : j ≠ l) :
    IndepFun (f i * f j) (f k * f l) κ μ :=
  (hf_indep.indepFun_prod_mk_prod_mk hf_meas i j k l hik hil hjk hjl).comp
    measurable_mul measurable_mul


@[to_additive]
lemma iIndepFun.indepFun_div_left (hf_indep : iIndepFun (fun _ ↦ m) f κ μ)
    (hf_meas : ∀ i, Measurable (f i)) (i j k : ι) (hik : i ≠ k) (hjk : j ≠ k) :
    IndepFun (f i / f j) (f k) κ μ := by
  have : IndepFun (fun ω => (f i ω, f j ω)) (f k) κ μ :=
    hf_indep.indepFun_prod_mk hf_meas i j k hik hjk
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    _mΩ : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μ : MeasureTheory.Measure α
    β : Type u_8
    m : MeasurableSpace β
    inst✝¹ : Div β
    inst✝ : MeasurableDiv₂ β
    f : ι → Ω → β
    hf_indep : ProbabilityTheory.Kernel.iIndepFun (fun x => m) f κ μ
    hf_meas : ∀ (i : ι), Measurable (f i)
    i j k : ι
    hik : Ne i k
    hjk : Ne j k
    this : ProbabilityTheory.Kernel.IndepFun (fun ω => { fst := f i ω, snd := f j  …
    ⊢ ProbabilityTheory.Kernel.IndepFun (HDiv.hDiv (f i) (f j)) (f k) κ μ
  -/
  simpa using this.comp (measurable_fst.div measurable_snd) measurable_id
  /-
    🎉 no goals
  -/


@[to_additive]
lemma iIndepFun.indepFun_div_right (hf_indep : iIndepFun (fun _ ↦ m) f κ μ)
    (hf_meas : ∀ i, Measurable (f i)) (i j k : ι) (hij : i ≠ j) (hik : i ≠ k) :
    IndepFun (f i) (f j / f k) κ μ :=
  (hf_indep.indepFun_div_left hf_meas _ _ _ hij.symm hik.symm).symm


@[to_additive]
lemma iIndepFun.indepFun_div_div (hf_indep : iIndepFun (fun _ ↦ m) f κ μ)
    (hf_meas : ∀ i, Measurable (f i))
    (i j k l : ι) (hik : i ≠ k) (hil : i ≠ l) (hjk : j ≠ k) (hjl : j ≠ l) :
    IndepFun (f i / f j) (f k / f l) κ μ :=
  (hf_indep.indepFun_prod_mk_prod_mk hf_meas i j k l hik hil hjk hjl).comp
    measurable_div measurable_div


@[to_additive]
theorem iIndepFun.indepFun_finset_prod_of_not_mem (hf_Indep : iIndepFun (fun _ ↦ m) f κ μ)
    (hf_meas : ∀ i, Measurable (f i)) {s : Finset ι} {i : ι} (hi : i ∉ s) :
    IndepFun (∏ j ∈ s, f j) (f i) κ μ := by
  classical
  have h_right : f i =
    (fun p : ({i} : Finset ι) → β => p ⟨i, Finset.mem_singleton_self i⟩) ∘
    fun a (j : ({i} : Finset ι)) => f j a := rfl
  have h_meas_right : Measurable fun p : ({i} : Finset ι) → β =>
      p ⟨i, Finset.mem_singleton_self i⟩ := measurable_pi_apply _
  have h_left : ∏ j ∈ s, f j = (fun p : s → β => ∏ j, p j) ∘ fun a (j : s) => f j a := by
    ext1 a
    simp only [Function.comp_apply]
    have : (∏ j : ↥s, f (↑j) a) = (∏ j : ↥s, f ↑j) a := by rw [Finset.prod_apply]
    rw [this, Finset.prod_coe_sort]
  have h_meas_left : Measurable fun p : s → β => ∏ j, p j :=
    Finset.univ.measurable_prod fun (j : ↥s) (_H : j ∈ Finset.univ) => measurable_pi_apply j
  rw [h_left, h_right]
  exact
    (hf_Indep.indepFun_finset s {i} (Finset.disjoint_singleton_left.mpr hi).symm hf_meas).comp
      h_meas_left h_meas_right


@[to_additive]
theorem iIndepFun.indepFun_prod_range_succ {f : ℕ → Ω → β}
    (hf_Indep : iIndepFun (fun _ => m) f κ μ) (hf_meas : ∀ i, Measurable (f i)) (n : ℕ) :
    IndepFun (∏ j ∈ Finset.range n, f j) (f n) κ μ :=
  hf_Indep.indepFun_finset_prod_of_not_mem hf_meas Finset.not_mem_range_self


theorem iIndepSet.iIndepFun_indicator [Zero β] [One β] {m : MeasurableSpace β} {s : ι → Set Ω}
    (hs : iIndepSet s κ μ) :
    iIndepFun (fun _n => m) (fun n => (s n).indicator fun _ω => 1) κ μ := by
  classical
  rw [iIndepFun_iff_measure_inter_preimage_eq_mul]
  rintro S π _hπ
  simp_rw [Set.indicator_const_preimage_eq_union]
  refine @hs S (fun i => ite (1 ∈ π i) (s i) ∅ ∪ ite ((0 : β) ∈ π i) (s i)ᶜ ∅) fun i _hi => ?_
  have hsi : MeasurableSet[generateFrom {s i}] (s i) :=
    measurableSet_generateFrom (Set.mem_singleton _)
  refine
    MeasurableSet.union (MeasurableSet.ite' (fun _ => hsi) fun _ => ?_)
      (MeasurableSet.ite' (fun _ => hsi.compl) fun _ => ?_)
  · exact @MeasurableSet.empty _ (generateFrom {s i})
  · exact @MeasurableSet.empty _ (generateFrom {s i})


/-- The probability of an intersection of preimages conditioning on another intersection factors
into a product. -/
lemma iIndepFun.cond_iInter [Finite ι] (hY : ∀ i, Measurable (Y i))
    (hindep : iIndepFun (fun _ ↦ mα.prod mβ) (fun i ω ↦ (X i ω, Y i ω)) κ μ)
    (hf : ∀ i ∈ s, MeasurableSet[mα.comap (X i)] (f i))
    (hy : ∀ᵐ a ∂μ, ∀ i ∉ s, κ a (Y i ⁻¹' t i) ≠ 0) (ht : ∀ i, MeasurableSet (t i)) :
    ∀ᵐ a ∂μ, (κ a)[⋂ i ∈ s, f i | ⋂ i, Y i ⁻¹' t i] = ∏ i ∈ s, (κ a)[f i | Y i in t i] := by
  classical
  cases nonempty_fintype ι
  let g (i' : ι) := if i' ∈ s then Y i' ⁻¹' t i' ∩ f i' else Y i' ⁻¹' t i'
  have hYt i : MeasurableSet[(mα.prod mβ).comap fun ω ↦ (X i ω, Y i ω)] (Y i ⁻¹' t i) :=
    ⟨.univ ×ˢ t i, .prod .univ (ht _), by ext; simp [eq_comm]⟩
  have hg i : MeasurableSet[(mα.prod mβ).comap fun ω ↦ (X i ω, Y i ω)] (g i) := by
    by_cases hi : i ∈ s <;> simp only [hi, ↓reduceIte, g]
    · obtain ⟨A, hA, hA'⟩ := hf i hi
      exact (hYt _).inter ⟨A ×ˢ .univ, hA.prod .univ, by ext; simp [← hA']⟩
    · exact hYt _
  filter_upwards [hy, hindep.ae_isProbabilityMeasure, hindep.meas_iInter hYt, hindep.meas_iInter hg]
    with a hy _ hYt hg
  calc
    _ = (κ a (⋂ i, Y i ⁻¹' t i))⁻¹ * κ a ((⋂ i, Y i ⁻¹' t i) ∩ ⋂ i ∈ s, f i) := by
      rw [cond_apply]; exact .iInter fun i ↦ hY i (ht i)
    _ = (κ a (⋂ i, Y i ⁻¹' t i))⁻¹ * κ a (⋂ i, g i) := by
      congr
      calc
        _ = (⋂ i, Y i ⁻¹' t i) ∩ ⋂ i, if i ∈ s then f i else .univ := by
          congr
          simp only [Set.iInter_ite, Set.iInter_univ, Set.inter_univ]
        _ = ⋂ i, Y i ⁻¹' t i ∩ (if i ∈ s then f i else .univ) := by rw [Set.iInter_inter_distrib]
        _ = _ := Set.iInter_congr fun i ↦ by by_cases hi : i ∈ s <;> simp [hi, g]
    _ = (∏ i, κ a (Y i ⁻¹' t i))⁻¹ * κ a (⋂ i, g i) := by
      rw [hYt]
    _ = (∏ i, κ a (Y i ⁻¹' t i))⁻¹ * ∏ i, κ a (g i) := by
      rw [hg]
    _ = ∏ i, (κ a (Y i ⁻¹' t i))⁻¹ * κ a (g i) := by
      rw [Finset.prod_mul_distrib, ENNReal.prod_inv_distrib]
      exact fun _ _ i _ _ ↦ .inr <| measure_ne_top _ _
    _ = ∏ i, if i ∈ s then (κ a)[f i | Y i ⁻¹' t i] else 1 := by
      refine Finset.prod_congr rfl fun i _ ↦ ?_
      by_cases hi : i ∈ s
      · simp only [hi, ↓reduceIte, g, cond_apply (hY i (ht i))]
      · simp only [hi, ↓reduceIte, g, ENNReal.inv_mul_cancel (hy i hi) (measure_ne_top _ _)]
    _ = _ := by simp

-- TODO: We can't state `Kernel.iIndepFun.cond` (the `Kernel` analogue of
-- `ProbabilityTheory.iIndepFun.cond`) because we don't have a version of `ProbabilityTheory.cond`
-- for kernels


