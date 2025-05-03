/-- A space is completely regular if points can be separated from closed sets via
  continuous functions to the unit interval. -/
@[mk_iff]
class CompletelyRegularSpace (X : Type u) [TopologicalSpace X] : Prop where
  completely_regular : ∀ (x : X), ∀ K : Set X, IsClosed K → x ∉ K →
    ∃ f : X → I, Continuous f ∧ f x = 0 ∧ EqOn f 1 K


instance CompletelyRegularSpace.instRegularSpace [CompletelyRegularSpace X] : RegularSpace X := by
  /-
    X : Type u
    inst✝² : TopologicalSpace X
    inst✝¹ : T1Space X
    inst✝ : CompletelyRegularSpace X
    ⊢ RegularSpace X
  -/
  rw [regularSpace_iff]
  /-
    X : Type u
    inst✝² : TopologicalSpace X
    inst✝¹ : T1Space X
    inst✝ : CompletelyRegularSpace X
    ⊢ ∀ {s : Set X} {a : X}, IsClosed s → Not (Membership.mem s a) → Disjoint (nhd …
  -/
  intro s a hs ha
  /-
    X : Type u
    inst✝² : TopologicalSpace X
    inst✝¹ : T1Space X
    inst✝ : CompletelyRegularSpace X
    s : Set X
    a : X
    hs : IsClosed s
    ha : Not (Membership.mem s a)
    ⊢ Disjoint (nhdsSet s) (nhds a)
  -/
  obtain ⟨f, cf, hf, hhf⟩ := CompletelyRegularSpace.completely_regular a s hs ha
  /-
    case intro.intro.intro
    X : Type u
    inst✝² : TopologicalSpace X
    inst✝¹ : T1Space X
    inst✝ : CompletelyRegularSpace X
    s : Set X
    a : X
    hs : IsClosed s
    ha : Not (Membership.mem s a)
    f : X → ↑unitInterval
    cf : Continuous f
    hf : Eq (f a) 0
    hhf : Set.EqOn f 1 s
    ⊢ Disjoint (nhdsSet s) (nhds a)
  -/
  apply disjoint_of_map (f := f)
  /-
    case intro.intro.intro
    X : Type u
    inst✝² : TopologicalSpace X
    inst✝¹ : T1Space X
    inst✝ : CompletelyRegularSpace X
    s : Set X
    a : X
    hs : IsClosed s
    ha : Not (Membership.mem s a)
    f : X → ↑unitInterval
    cf : Continuous f
    hf : Eq (f a) 0
    hhf : Set.EqOn f 1 s
    ⊢ Disjoint (Filter.map f (nhdsSet s)) (Filter.map f (nhds a))
  -/
  apply Disjoint.mono (cf.tendsto_nhdsSet_nhds hhf) cf.continuousAt
  /-
    case intro.intro.intro
    X : Type u
    inst✝² : TopologicalSpace X
    inst✝¹ : T1Space X
    inst✝ : CompletelyRegularSpace X
    s : Set X
    a : X
    hs : IsClosed s
    ha : Not (Membership.mem s a)
    f : X → ↑unitInterval
    cf : Continuous f
    hf : Eq (f a) 0
    hhf : Set.EqOn f 1 s
    ⊢ Disjoint (nhds ⟨1, unitInterval.hasOne.proof_1⟩) (nhds (f a))
  -/
  exact disjoint_nhds_nhds.mpr (hf.symm ▸ zero_ne_one).symm
  /-
    🎉 no goals
  -/


instance NormalSpace.instCompletelyRegularSpace [NormalSpace X] : CompletelyRegularSpace X := by
  /-
    X : Type u
    inst✝² : TopologicalSpace X
    inst✝¹ : T1Space X
    inst✝ : NormalSpace X
    ⊢ CompletelyRegularSpace X
  -/
  rw [completelyRegularSpace_iff]
  /-
    X : Type u
    inst✝² : TopologicalSpace X
    inst✝¹ : T1Space X
    inst✝ : NormalSpace X
    ⊢ ∀ (x : X) (K : Set X), IsClosed K → Not (Membership.mem K x) → Exists fun f  …
  -/
  intro x K hK hx
  /-
    X : Type u
    inst✝² : TopologicalSpace X
    inst✝¹ : T1Space X
    inst✝ : NormalSpace X
    x : X
    K : Set X
    hK : IsClosed K
    hx : Not (Membership.mem K x)
    ⊢ Exists fun f => And (Continuous f) (And (Eq (f x) 0) (Set.EqOn f 1 K))
  -/
  have cx : IsClosed {x} := T1Space.t1 x
  /-
    X : Type u
    inst✝² : TopologicalSpace X
    inst✝¹ : T1Space X
    inst✝ : NormalSpace X
    x : X
    K : Set X
    hK : IsClosed K
    hx : Not (Membership.mem K x)
    cx : IsClosed (Singleton.singleton x)
    ⊢ Exists fun f => And (Continuous f) (And (Eq (f x) 0) (Set.EqOn f 1 K))
  -/
  have d : Disjoint {x} K := by rwa [Set.disjoint_iff, subset_empty_iff, singleton_inter_eq_empty]
  /-
    X : Type u
    inst✝² : TopologicalSpace X
    inst✝¹ : T1Space X
    inst✝ : NormalSpace X
    x : X
    K : Set X
    hK : IsClosed K
    hx : Not (Membership.mem K x)
    cx : IsClosed (Singleton.singleton x)
    d : Disjoint (Singleton.singleton x) K
    ⊢ Exists fun f => And (Continuous f) (And (Eq (f x) 0) (Set.EqOn f 1 K))
  -/
  let ⟨⟨f, cf⟩, hfx, hfK, hficc⟩ := exists_continuous_zero_one_of_isClosed cx hK d
  /-
    X : Type u
    inst✝² : TopologicalSpace X
    inst✝¹ : T1Space X
    inst✝ : NormalSpace X
    x : X
    K : Set X
    hK : IsClosed K
    hx : Not (Membership.mem K x)
    cx : IsClosed (Singleton.singleton x)
    d : Disjoint (Singleton.singleton x) K
    f : X → Real
    cf : Continuous f
    hfx : Set.EqOn (⇑{ toFun := f, continuous_toFun := cf }) 0 (Singleton.singleto …
    hfK : Set.EqOn (⇑{ toFun := f, continuous_toFun := cf }) 1 K
    hficc : ∀ (x : X), Membership.mem (Set.Icc 0 1) ({ toFun := f, continuous_toFu …
    ⊢ Exists fun f => And (Continuous f) (And (Eq (f x) 0) (Set.EqOn f 1 K))
  -/
  let g : X → I := fun x => ⟨f x, hficc x⟩
  /-
    X : Type u
    inst✝² : TopologicalSpace X
    inst✝¹ : T1Space X
    inst✝ : NormalSpace X
    x : X
    K : Set X
    hK : IsClosed K
    hx : Not (Membership.mem K x)
    cx : IsClosed (Singleton.singleton x)
    d : Disjoint (Singleton.singleton x) K
    f : X → Real
    cf : Continuous f
    hfx : Set.EqOn (⇑{ toFun := f, continuous_toFun := cf }) 0 (Singleton.singleto …
    hfK : Set.EqOn (⇑{ toFun := f, continuous_toFun := cf }) 1 K
    hficc : ∀ (x : X), Membership.mem (Set.Icc 0 1) ({ toFun := f, continuous_toFu …
    g : X → ↑unitInterval := fun x => ⟨f x, ⋯⟩
    ⊢ Exists fun f => And (Continuous f) (And (Eq (f x) 0) (Set.EqOn f 1 K))
  -/
  have cg : Continuous g := cf.subtype_mk hficc
  /-
    X : Type u
    inst✝² : TopologicalSpace X
    inst✝¹ : T1Space X
    inst✝ : NormalSpace X
    x : X
    K : Set X
    hK : IsClosed K
    hx : Not (Membership.mem K x)
    cx : IsClosed (Singleton.singleton x)
    d : Disjoint (Singleton.singleton x) K
    f : X → Real
    cf : Continuous f
    hfx : Set.EqOn (⇑{ toFun := f, continuous_toFun := cf }) 0 (Singleton.singleto …
    hfK : Set.EqOn (⇑{ toFun := f, continuous_toFun := cf }) 1 K
    hficc : ∀ (x : X), Membership.mem (Set.Icc 0 1) ({ toFun := f, continuous_toFu …
    g : X → ↑unitInterval := fun x => ⟨f x, ⋯⟩
    cg : Continuous g
    ⊢ Exists fun f => And (Continuous f) (And (Eq (f x) 0) (Set.EqOn f 1 K))
  -/
  have hgx : g x = 0 := Subtype.ext (hfx (mem_singleton_iff.mpr (Eq.refl x)))
  /-
    X : Type u
    inst✝² : TopologicalSpace X
    inst✝¹ : T1Space X
    inst✝ : NormalSpace X
    x : X
    K : Set X
    hK : IsClosed K
    hx : Not (Membership.mem K x)
    cx : IsClosed (Singleton.singleton x)
    d : Disjoint (Singleton.singleton x) K
    f : X → Real
    cf : Continuous f
    hfx : Set.EqOn (⇑{ toFun := f, continuous_toFun := cf }) 0 (Singleton.singleto …
    hfK : Set.EqOn (⇑{ toFun := f, continuous_toFun := cf }) 1 K
    hficc : ∀ (x : X), Membership.mem (Set.Icc 0 1) ({ toFun := f, continuous_toFu …
    g : X → ↑unitInterval := fun x => ⟨f x, ⋯⟩
    cg : Continuous g
    hgx : Eq (g x) 0
    ⊢ Exists fun f => And (Continuous f) (And (Eq (f x) 0) (Set.EqOn f 1 K))
  -/
  have hgK : EqOn g 1 K := fun k hk => Subtype.ext (hfK hk)
  /-
    X : Type u
    inst✝² : TopologicalSpace X
    inst✝¹ : T1Space X
    inst✝ : NormalSpace X
    x : X
    K : Set X
    hK : IsClosed K
    hx : Not (Membership.mem K x)
    cx : IsClosed (Singleton.singleton x)
    d : Disjoint (Singleton.singleton x) K
    f : X → Real
    cf : Continuous f
    hfx : Set.EqOn (⇑{ toFun := f, continuous_toFun := cf }) 0 (Singleton.singleto …
    hfK : Set.EqOn (⇑{ toFun := f, continuous_toFun := cf }) 1 K
    hficc : ∀ (x : X), Membership.mem (Set.Icc 0 1) ({ toFun := f, continuous_toFu …
    g : X → ↑unitInterval := fun x => ⟨f x, ⋯⟩
    cg : Continuous g
    hgx : Eq (g x) 0
    hgK : Set.EqOn g 1 K
    ⊢ Exists fun f => And (Continuous f) (And (Eq (f x) 0) (Set.EqOn f 1 K))
  -/
  exact ⟨g, cg, hgx, hgK⟩
  /-
    🎉 no goals
  -/


/-- A T₃.₅ space is a completely regular space that is also T1. -/
@[mk_iff]
class T35Space (X : Type u) [TopologicalSpace X] extends T1Space X, CompletelyRegularSpace X : Prop


instance T35Space.instT3space [T35Space X] : T3Space X := {}


instance T4Space.instT35Space [T4Space X] : T35Space X := {}


lemma separatesPoints_continuous_of_t35Space [T35Space X] :
    SeparatesPoints (Continuous : Set (X → ℝ)) := by
  /-
    X : Type u
    inst✝² : TopologicalSpace X
    inst✝¹ : T1Space X
    inst✝ : T35Space X
    ⊢ Set.SeparatesPoints Continuous
  -/
  intro x y x_ne_y
  obtain ⟨f, f_cont, f_zero, f_one⟩ :=
    CompletelyRegularSpace.completely_regular x {y} isClosed_singleton x_ne_y
  /-
    case intro.intro.intro
    X : Type u
    inst✝² : TopologicalSpace X
    inst✝¹ : T1Space X
    inst✝ : T35Space X
    x y : X
    x_ne_y : Ne x y
    f : X → ↑unitInterval
    f_cont : Continuous f
    f_zero : Eq (f x) 0
    f_one : Set.EqOn f 1 (Singleton.singleton y)
    ⊢ Exists fun f => And (Membership.mem Continuous f) (Ne (f x) (f y))
  -/
  exact ⟨fun x ↦ f x, continuous_subtype_val.comp f_cont, by aesop⟩
  /-
    🎉 no goals
  -/


lemma separatesPoints_continuous_of_t35Space_Icc [T35Space X] :
    SeparatesPoints (Continuous : Set (X → I)) := by
  /-
    X : Type u
    inst✝² : TopologicalSpace X
    inst✝¹ : T1Space X
    inst✝ : T35Space X
    ⊢ Set.SeparatesPoints Continuous
  -/
  intro x y x_ne_y
  obtain ⟨f, f_cont, f_zero, f_one⟩ :=
    CompletelyRegularSpace.completely_regular x {y} isClosed_singleton x_ne_y
  /-
    case intro.intro.intro
    X : Type u
    inst✝² : TopologicalSpace X
    inst✝¹ : T1Space X
    inst✝ : T35Space X
    x y : X
    x_ne_y : Ne x y
    f : X → ↑unitInterval
    f_cont : Continuous f
    f_zero : Eq (f x) 0
    f_one : Set.EqOn f 1 (Singleton.singleton y)
    ⊢ Exists fun f => And (Membership.mem Continuous f) (Ne (f x) (f y))
  -/
  exact ⟨f, f_cont, by aesop⟩
  /-
    🎉 no goals
  -/


lemma injective_stoneCechUnit_of_t35Space [T35Space X] :
    Function.Injective (stoneCechUnit : X → StoneCech X) := by
  /-
    X : Type u
    inst✝² : TopologicalSpace X
    inst✝¹ : T1Space X
    inst✝ : T35Space X
    ⊢ Function.Injective stoneCechUnit
  -/
  intros a b hab
  /-
    X : Type u
    inst✝² : TopologicalSpace X
    inst✝¹ : T1Space X
    inst✝ : T35Space X
    a b : X
    hab : Eq (stoneCechUnit a) (stoneCechUnit b)
    ⊢ Eq a b
  -/
  contrapose hab
  /-
    X : Type u
    inst✝² : TopologicalSpace X
    inst✝¹ : T1Space X
    inst✝ : T35Space X
    a b : X
    hab : Not (Eq a b)
    ⊢ Not (Eq (stoneCechUnit a) (stoneCechUnit b))
  -/
  obtain ⟨f, fc, fab⟩ := separatesPoints_continuous_of_t35Space_Icc hab
  /-
    case intro.intro
    X : Type u
    inst✝² : TopologicalSpace X
    inst✝¹ : T1Space X
    inst✝ : T35Space X
    a b : X
    hab : Not (Eq a b)
    f : X → ↑unitInterval
    fc : Membership.mem Continuous f
    fab : Ne (f a) (f b)
    ⊢ Not (Eq (stoneCechUnit a) (stoneCechUnit b))
  -/
  exact fun q ↦ fab (eq_if_stoneCechUnit_eq fc q)
  /-
    🎉 no goals
  -/

