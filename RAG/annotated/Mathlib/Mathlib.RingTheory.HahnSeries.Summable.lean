/-- A family of Hahn series whose formal coefficient-wise sum is a Hahn series.  For each
coefficient of the sum to be well-defined, we require that only finitely many series are nonzero at
any given coefficient.  For the formal sum to be a Hahn series, we require that the union of the
supports of the constituent series is partially well-ordered. -/
structure SummableFamily (Γ) (R) [PartialOrder Γ] [AddCommMonoid R] (α : Type*) where
  /-- A parametrized family of Hahn series. -/
  toFun : α → HahnSeries Γ R
  isPWO_iUnion_support' : Set.IsPWO (⋃ a : α, (toFun a).support)
  finite_co_support' : ∀ g : Γ, { a | (toFun a).coeff g ≠ 0 }.Finite


instance : FunLike (SummableFamily Γ R α) α (HahnSeries Γ R) where
  coe := toFun
  coe_injective' | ⟨_, _, _⟩, ⟨_, _, _⟩, rfl => rfl


theorem isPWO_iUnion_support (s : SummableFamily Γ R α) : Set.IsPWO (⋃ a : α, (s a).support) :=
  s.isPWO_iUnion_support'


theorem finite_co_support (s : SummableFamily Γ R α) (g : Γ) :
    (Function.support fun a => (s a).coeff g).Finite :=
  s.finite_co_support' g


theorem coe_injective : @Function.Injective (SummableFamily Γ R α) (α → HahnSeries Γ R) (⇑) :=
  DFunLike.coe_injective


@[ext]
theorem ext {s t : SummableFamily Γ R α} (h : ∀ a : α, s a = t a) : s = t :=
  DFunLike.ext s t h


instance : Add (SummableFamily Γ R α) :=
  ⟨fun x y =>
    { toFun := x + y
      isPWO_iUnion_support' :=
        (x.isPWO_iUnion_support.union y.isPWO_iUnion_support).mono
          (by
            /-
              Γ : Type u_1
              Γ' : Type u_2
              R : Type u_3
              V : Type u_4
              α : Type u_5
              β : Type u_6
              inst✝¹ : PartialOrder Γ
              inst✝ : AddCommMonoid R
              x y : HahnSeries.SummableFamily Γ R α
              ⊢ HasSubset.Subset (Set.iUnion fun a => (HAdd.hAdd (⇑x) (⇑y) a).support) (Unio …
            -/
            rw [← Set.iUnion_union_distrib]
            /-
              Γ : Type u_1
              Γ' : Type u_2
              R : Type u_3
              V : Type u_4
              α : Type u_5
              β : Type u_6
              inst✝¹ : PartialOrder Γ
              inst✝ : AddCommMonoid R
              x y : HahnSeries.SummableFamily Γ R α
              ⊢ HasSubset.Subset (Set.iUnion fun a => (HAdd.hAdd (⇑x) (⇑y) a).support) (Set. …
            -/
            exact Set.iUnion_mono fun a => support_add_subset)
            /-
              🎉 no goals
            -/
      finite_co_support' := fun g =>
        ((x.finite_co_support g).union (y.finite_co_support g)).subset
          (by
            /-
              Γ : Type u_1
              Γ' : Type u_2
              R : Type u_3
              V : Type u_4
              α : Type u_5
              β : Type u_6
              inst✝¹ : PartialOrder Γ
              inst✝ : AddCommMonoid R
              x y : HahnSeries.SummableFamily Γ R α
              g : Γ
              ⊢ HasSubset.Subset (setOf fun a => Ne ((HAdd.hAdd (⇑x) (⇑y) a).coeff g) 0) (Un …
            -/
            intro a ha
            /-
              Γ : Type u_1
              Γ' : Type u_2
              R : Type u_3
              V : Type u_4
              α : Type u_5
              β : Type u_6
              inst✝¹ : PartialOrder Γ
              inst✝ : AddCommMonoid R
              x y : HahnSeries.SummableFamily Γ R α
              g : Γ
              a : α
              ha : Membership.mem (setOf fun a => Ne ((HAdd.hAdd (⇑x) (⇑y) a).coeff g) 0) a
              ⊢ Membership.mem (Union.union (Function.support fun a => (x a).coeff g) (Funct …
            -/
            change (x a).coeff g + (y a).coeff g ≠ 0 at ha
            /-
              Γ : Type u_1
              Γ' : Type u_2
              R : Type u_3
              V : Type u_4
              α : Type u_5
              β : Type u_6
              inst✝¹ : PartialOrder Γ
              inst✝ : AddCommMonoid R
              x y : HahnSeries.SummableFamily Γ R α
              g : Γ
              a : α
              ha : Ne (HAdd.hAdd ((x a).coeff g) ((y a).coeff g)) 0
              ⊢ Membership.mem (Union.union (Function.support fun a => (x a).coeff g) (Funct …
            -/
            rw [Set.mem_union, Function.mem_support, Function.mem_support]
            /-
              Γ : Type u_1
              Γ' : Type u_2
              R : Type u_3
              V : Type u_4
              α : Type u_5
              β : Type u_6
              inst✝¹ : PartialOrder Γ
              inst✝ : AddCommMonoid R
              x y : HahnSeries.SummableFamily Γ R α
              g : Γ
              a : α
              ha : Ne (HAdd.hAdd ((x a).coeff g) ((y a).coeff g)) 0
              ⊢ Or (Ne ((x a).coeff g) 0) (Ne ((y a).coeff g) 0)
            -/
            contrapose! ha
            /-
              Γ : Type u_1
              Γ' : Type u_2
              R : Type u_3
              V : Type u_4
              α : Type u_5
              β : Type u_6
              inst✝¹ : PartialOrder Γ
              inst✝ : AddCommMonoid R
              x y : HahnSeries.SummableFamily Γ R α
              g : Γ
              a : α
              ha : And (Eq ((x a).coeff g) 0) (Eq ((y a).coeff g) 0)
              ⊢ Eq (HAdd.hAdd ((x a).coeff g) ((y a).coeff g)) 0
            -/
            rw [ha.1, ha.2, add_zero]) }⟩
            /-
              🎉 no goals
            -/


instance : Zero (SummableFamily Γ R α) :=
          /-
            Γ : Type u_1
            Γ' : Type u_2
            R : Type u_3
            V : Type u_4
            α : Type u_5
            β : Type u_6
            inst✝¹ : PartialOrder Γ
            inst✝ : AddCommMonoid R
            ⊢ (Set.iUnion fun a => (0 a).support).IsPWO
          -/
          /-
            🎉 no goals
          -/
  ⟨⟨0, by simp, by simp⟩⟩
                   /-
                     🎉 no goals
                   -/


instance : Inhabited (SummableFamily Γ R α) :=
  ⟨0⟩


@[simp]
theorem coe_add {s t : SummableFamily Γ R α} : ⇑(s + t) = s + t :=
  rfl


theorem add_apply {s t : SummableFamily Γ R α} {a : α} : (s + t) a = s a + t a :=
  rfl


@[simp]
theorem coe_zero : ((0 : SummableFamily Γ R α) : α → HahnSeries Γ R) = 0 :=
  rfl


theorem zero_apply {a : α} : (0 : SummableFamily Γ R α) a = 0 :=
  rfl


instance : AddCommMonoid (SummableFamily Γ R α) where
  zero := 0
  nsmul := nsmulRec
  zero_add s := by
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_4
      α : Type u_5
      β : Type u_6
      inst✝¹ : PartialOrder Γ
      inst✝ : AddCommMonoid R
      s : HahnSeries.SummableFamily Γ R α
      ⊢ Eq (HAdd.hAdd 0 s) s
    -/
    ext
    /-
      case h.coeff.h
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_4
      α : Type u_5
      β : Type u_6
      inst✝¹ : PartialOrder Γ
      inst✝ : AddCommMonoid R
      s : HahnSeries.SummableFamily Γ R α
      a✝ : α
      x✝ : Γ
      ⊢ Eq (((HAdd.hAdd 0 s) a✝).coeff x✝) ((s a✝).coeff x✝)
    -/
    apply zero_add
    /-
      🎉 no goals
    -/
  add_zero s := by
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_4
      α : Type u_5
      β : Type u_6
      inst✝¹ : PartialOrder Γ
      inst✝ : AddCommMonoid R
      s : HahnSeries.SummableFamily Γ R α
      ⊢ Eq (HAdd.hAdd s 0) s
    -/
    ext
    /-
      case h.coeff.h
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_4
      α : Type u_5
      β : Type u_6
      inst✝¹ : PartialOrder Γ
      inst✝ : AddCommMonoid R
      s : HahnSeries.SummableFamily Γ R α
      a✝ : α
      x✝ : Γ
      ⊢ Eq (((HAdd.hAdd s 0) a✝).coeff x✝) ((s a✝).coeff x✝)
    -/
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_4
      α : Type u_5
      β : Type u_6
      inst✝¹ : PartialOrder Γ
      inst✝ : AddCommMonoid R
      r s t : HahnSeries.SummableFamily Γ R α
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd r s) t) (HAdd.hAdd r (HAdd.hAdd s t))
    -/
    apply add_zero
    /-
      case h.coeff.h
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_4
      α : Type u_5
      β : Type u_6
      inst✝¹ : PartialOrder Γ
      inst✝ : AddCommMonoid R
      r s t : HahnSeries.SummableFamily Γ R α
      a✝ : α
      x✝ : Γ
      ⊢ Eq (((HAdd.hAdd (HAdd.hAdd r s) t) a✝).coeff x✝) (((HAdd.hAdd r (HAdd.hAdd s …
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
  add_comm s t := by
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_4
      α : Type u_5
      β : Type u_6
      inst✝¹ : PartialOrder Γ
      inst✝ : AddCommMonoid R
      s t : HahnSeries.SummableFamily Γ R α
      ⊢ Eq (HAdd.hAdd s t) (HAdd.hAdd t s)
    -/
    ext
    /-
      case h.coeff.h
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_4
      α : Type u_5
      β : Type u_6
      inst✝¹ : PartialOrder Γ
      inst✝ : AddCommMonoid R
      s t : HahnSeries.SummableFamily Γ R α
      a✝ : α
      x✝ : Γ
      ⊢ Eq (((HAdd.hAdd s t) a✝).coeff x✝) (((HAdd.hAdd t s) a✝).coeff x✝)
    -/
    apply add_comm
    /-
      🎉 no goals
    -/
  add_assoc r s t := by
    ext
    apply add_assoc


/-- The coefficient function of a summable family, as a finsupp on the parameter type. -/
@[simps]
def coeff (s : SummableFamily Γ R α) (g : Γ) : α →₀ R where
  support := (s.finite_co_support g).toFinset
  toFun a := (s a).coeff g
                            /-
                              Γ : Type u_1
                              Γ' : Type u_2
                              R : Type u_3
                              V : Type u_4
                              α : Type u_5
                              β : Type u_6
                              inst✝¹ : PartialOrder Γ
                              inst✝ : AddCommMonoid R
                              s : HahnSeries.SummableFamily Γ R α
                              g : Γ
                              a : α
                              ⊢ Iff (Membership.mem ⋯.toFinset a) (Ne ((fun a => (s a).coeff g) a) 0)
                            -/
  mem_support_toFun a := by simp
                            /-
                              🎉 no goals
                            -/


@[simp]
theorem coeff_def (s : SummableFamily Γ R α) (a : α) (g : Γ) : s.coeff g a = (s a).coeff g :=
  rfl


/-- The infinite sum of a `SummableFamily` of Hahn series. -/
def hsum (s : SummableFamily Γ R α) : HahnSeries Γ R where
  coeff g := ∑ᶠ i, (s i).coeff g
  isPWO_support' :=
    s.isPWO_iUnion_support.mono fun g => by
      /-
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        V : Type u_4
        α : Type u_5
        β : Type u_6
        inst✝¹ : PartialOrder Γ
        inst✝ : AddCommMonoid R
        s : HahnSeries.SummableFamily Γ R α
        g : Γ
        ⊢ Membership.mem (Function.support fun g => finsum fun i => (s i).coeff g) g → …
      -/
      contrapose
      /-
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        V : Type u_4
        α : Type u_5
        β : Type u_6
        inst✝¹ : PartialOrder Γ
        inst✝ : AddCommMonoid R
        s : HahnSeries.SummableFamily Γ R α
        g : Γ
        ⊢ Not (Membership.mem (Set.iUnion fun a => (s a).support) g) → Not (Membership …
      -/
      rw [Set.mem_iUnion, not_exists, Function.mem_support, Classical.not_not]
      /-
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        V : Type u_4
        α : Type u_5
        β : Type u_6
        inst✝¹ : PartialOrder Γ
        inst✝ : AddCommMonoid R
        s : HahnSeries.SummableFamily Γ R α
        g : Γ
        ⊢ (∀ (x : α), Not (Membership.mem (s x).support g)) → Eq (finsum fun i => (s i …
      -/
      simp_rw [mem_support, Classical.not_not]
      /-
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        V : Type u_4
        α : Type u_5
        β : Type u_6
        inst✝¹ : PartialOrder Γ
        inst✝ : AddCommMonoid R
        s : HahnSeries.SummableFamily Γ R α
        g : Γ
        ⊢ (∀ (x : α), Eq ((s x).coeff g) 0) → Eq (finsum fun i => (s i).coeff g) 0
      -/
      intro h
      /-
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        V : Type u_4
        α : Type u_5
        β : Type u_6
        inst✝¹ : PartialOrder Γ
        inst✝ : AddCommMonoid R
        s : HahnSeries.SummableFamily Γ R α
        g : Γ
        h : ∀ (x : α), Eq ((s x).coeff g) 0
        ⊢ Eq (finsum fun i => (s i).coeff g) 0
      -/
      rw [finsum_congr h, finsum_zero]
      /-
        🎉 no goals
      -/


@[simp]
theorem hsum_coeff {s : SummableFamily Γ R α} {g : Γ} : s.hsum.coeff g = ∑ᶠ i, (s i).coeff g :=
  rfl


theorem support_hsum_subset {s : SummableFamily Γ R α} : s.hsum.support ⊆ ⋃ a : α, (s a).support :=
  fun g hg => by
  /-
    Γ : Type u_1
    R : Type u_3
    α : Type u_5
    inst✝¹ : PartialOrder Γ
    inst✝ : AddCommMonoid R
    s : HahnSeries.SummableFamily Γ R α
    g : Γ
    hg : Membership.mem s.hsum.support g
    ⊢ Membership.mem (Set.iUnion fun a => (s a).support) g
  -/
  rw [mem_support, hsum_coeff, finsum_eq_sum _ (s.finite_co_support _)] at hg
  /-
    Γ : Type u_1
    R : Type u_3
    α : Type u_5
    inst✝¹ : PartialOrder Γ
    inst✝ : AddCommMonoid R
    s : HahnSeries.SummableFamily Γ R α
    g : Γ
    hg : Ne (⋯.toFinset.sum fun i => (s i).coeff g) 0
    ⊢ Membership.mem (Set.iUnion fun a => (s a).support) g
  -/
  obtain ⟨a, _, h2⟩ := exists_ne_zero_of_sum_ne_zero hg
  /-
    case intro.intro
    Γ : Type u_1
    R : Type u_3
    α : Type u_5
    inst✝¹ : PartialOrder Γ
    inst✝ : AddCommMonoid R
    s : HahnSeries.SummableFamily Γ R α
    g : Γ
    hg : Ne (⋯.toFinset.sum fun i => (s i).coeff g) 0
    a : α
    left✝ : Membership.mem ⋯.toFinset a
    h2 : Ne ((s a).coeff g) 0
    ⊢ Membership.mem (Set.iUnion fun a => (s a).support) g
  -/
  rw [Set.mem_iUnion]
  /-
    case intro.intro
    Γ : Type u_1
    R : Type u_3
    α : Type u_5
    inst✝¹ : PartialOrder Γ
    inst✝ : AddCommMonoid R
    s : HahnSeries.SummableFamily Γ R α
    g : Γ
    hg : Ne (⋯.toFinset.sum fun i => (s i).coeff g) 0
    a : α
    left✝ : Membership.mem ⋯.toFinset a
    h2 : Ne ((s a).coeff g) 0
    ⊢ Exists fun i => Membership.mem (s i).support g
  -/
  exact ⟨a, h2⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem hsum_add {s t : SummableFamily Γ R α} : (s + t).hsum = s.hsum + t.hsum := by
  /-
    Γ : Type u_1
    R : Type u_3
    α : Type u_5
    inst✝¹ : PartialOrder Γ
    inst✝ : AddCommMonoid R
    s t : HahnSeries.SummableFamily Γ R α
    ⊢ Eq (HAdd.hAdd s t).hsum (HAdd.hAdd s.hsum t.hsum)
  -/
  ext g
  /-
    case coeff.h
    Γ : Type u_1
    R : Type u_3
    α : Type u_5
    inst✝¹ : PartialOrder Γ
    inst✝ : AddCommMonoid R
    s t : HahnSeries.SummableFamily Γ R α
    g : Γ
    ⊢ Eq ((HAdd.hAdd s t).hsum.coeff g) ((HAdd.hAdd s.hsum t.hsum).coeff g)
  -/
  simp only [hsum_coeff, add_coeff, add_apply]
  /-
    case coeff.h
    Γ : Type u_1
    R : Type u_3
    α : Type u_5
    inst✝¹ : PartialOrder Γ
    inst✝ : AddCommMonoid R
    s t : HahnSeries.SummableFamily Γ R α
    g : Γ
    ⊢ Eq (finsum fun i => HAdd.hAdd ((s i).coeff g) ((t i).coeff g)) (HAdd.hAdd (f …
  -/
  exact finsum_add_distrib (s.finite_co_support _) (t.finite_co_support _)
  /-
    🎉 no goals
  -/


theorem hsum_coeff_eq_sum_of_subset {s : SummableFamily Γ R α} {g : Γ} {t : Finset α}
    (h : { a | (s a).coeff g ≠ 0 } ⊆ t) : s.hsum.coeff g = ∑ i ∈ t, (s i).coeff g := by
  /-
    Γ : Type u_1
    R : Type u_3
    α : Type u_5
    inst✝¹ : PartialOrder Γ
    inst✝ : AddCommMonoid R
    s : HahnSeries.SummableFamily Γ R α
    g : Γ
    t : Finset α
    h : HasSubset.Subset (setOf fun a => Ne ((s a).coeff g) 0) ↑t
    ⊢ Eq (s.hsum.coeff g) (t.sum fun i => (s i).coeff g)
  -/
  simp only [hsum_coeff, finsum_eq_sum _ (s.finite_co_support _)]
  /-
    Γ : Type u_1
    R : Type u_3
    α : Type u_5
    inst✝¹ : PartialOrder Γ
    inst✝ : AddCommMonoid R
    s : HahnSeries.SummableFamily Γ R α
    g : Γ
    t : Finset α
    h : HasSubset.Subset (setOf fun a => Ne ((s a).coeff g) 0) ↑t
    ⊢ Eq (⋯.toFinset.sum fun i => (s i).coeff g) (t.sum fun i => (s i).coeff g)
  -/
  exact sum_subset (Set.Finite.toFinset_subset.mpr h) (by simp)
  /-
    🎉 no goals
  -/


theorem hsum_coeff_eq_sum {s : SummableFamily Γ R α} {g : Γ} :
    s.hsum.coeff g = ∑ i ∈ (s.coeff g).support, (s i).coeff g := by
  /-
    Γ : Type u_1
    R : Type u_3
    α : Type u_5
    inst✝¹ : PartialOrder Γ
    inst✝ : AddCommMonoid R
    s : HahnSeries.SummableFamily Γ R α
    g : Γ
    ⊢ Eq (s.hsum.coeff g) ((s.coeff g).support.sum fun i => (s i).coeff g)
  -/
  simp only [hsum_coeff, finsum_eq_sum _ (s.finite_co_support _), coeff_support]
  /-
    🎉 no goals
  -/


/-- The summable family made of a single Hahn series. -/
@[simps]
def single (x : HahnSeries Γ R) : SummableFamily Γ R Unit where
  toFun _ := x
  isPWO_iUnion_support' :=
    Eq.mpr (congrArg (fun s ↦ s.IsPWO) (Set.iUnion_const x.support)) x.isPWO_support
  finite_co_support' g := Set.toFinite {a | ((fun _ ↦ x) a).coeff g ≠ 0}


@[simp]
theorem hsum_single (x : HahnSeries Γ R) : (single x).hsum = x := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : AddCommMonoid R
    x : HahnSeries Γ R
    ⊢ Eq (HahnSeries.SummableFamily.single x).hsum x
  -/
  ext g
  /-
    case coeff.h
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : AddCommMonoid R
    x : HahnSeries Γ R
    g : Γ
    ⊢ Eq ((HahnSeries.SummableFamily.single x).hsum.coeff g) (x.coeff g)
  -/
  simp only [hsum_coeff, single_toFun, finsum_unique]
  /-
    🎉 no goals
  -/


/-- A summable family induced by an equivalence of the parametrizing type. -/
@[simps]
def Equiv (e : α ≃ β) (s : SummableFamily Γ R α) : SummableFamily Γ R β where
  toFun b := s (e.symm b)
  isPWO_iUnion_support' := by
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_4
      α : Type u_5
      β : Type u_6
      inst✝¹ : PartialOrder Γ
      inst✝ : AddCommMonoid R
      e : _root_.Equiv α β
      s : HahnSeries.SummableFamily Γ R α
      ⊢ (Set.iUnion fun a => ((fun b => s (e.symm b)) a).support).IsPWO
    -/
    refine Set.IsPWO.mono s.isPWO_iUnion_support fun g => ?_
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_4
      α : Type u_5
      β : Type u_6
      inst✝¹ : PartialOrder Γ
      inst✝ : AddCommMonoid R
      e : _root_.Equiv α β
      s : HahnSeries.SummableFamily Γ R α
      g : Γ
      ⊢ Membership.mem (Set.iUnion fun a => ((fun b => s (e.symm b)) a).support) g → …
    -/
    simp only [Set.mem_iUnion, mem_support, ne_eq, forall_exists_index]
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_4
      α : Type u_5
      β : Type u_6
      inst✝¹ : PartialOrder Γ
      inst✝ : AddCommMonoid R
      e : _root_.Equiv α β
      s : HahnSeries.SummableFamily Γ R α
      g : Γ
      ⊢ ∀ (x : β), Not (Eq ((s (e.symm x)).coeff g) 0) → Exists fun i => Not (Eq ((s …
    -/
    exact fun b hg => Exists.intro (e.symm b) hg
    /-
      🎉 no goals
    -/
  finite_co_support' g :=
    (Equiv.set_finite_iff e.subtypeEquivOfSubtype').mp <| s.finite_co_support' g


@[simp]
theorem hsum_equiv (e : α ≃ β) (s : SummableFamily Γ R α) : (Equiv e s).hsum = s.hsum := by
  /-
    Γ : Type u_1
    R : Type u_3
    α : Type u_5
    β : Type u_6
    inst✝¹ : PartialOrder Γ
    inst✝ : AddCommMonoid R
    e : _root_.Equiv α β
    s : HahnSeries.SummableFamily Γ R α
    ⊢ Eq (HahnSeries.SummableFamily.Equiv e s).hsum s.hsum
  -/
  ext g
  /-
    case coeff.h
    Γ : Type u_1
    R : Type u_3
    α : Type u_5
    β : Type u_6
    inst✝¹ : PartialOrder Γ
    inst✝ : AddCommMonoid R
    e : _root_.Equiv α β
    s : HahnSeries.SummableFamily Γ R α
    g : Γ
    ⊢ Eq ((HahnSeries.SummableFamily.Equiv e s).hsum.coeff g) (s.hsum.coeff g)
  -/
  simp only [hsum_coeff, Equiv_toFun]
  /-
    case coeff.h
    Γ : Type u_1
    R : Type u_3
    α : Type u_5
    β : Type u_6
    inst✝¹ : PartialOrder Γ
    inst✝ : AddCommMonoid R
    e : _root_.Equiv α β
    s : HahnSeries.SummableFamily Γ R α
    g : Γ
    ⊢ Eq (finsum fun i => (s (e.symm i)).coeff g) (finsum fun i => (s i).coeff g)
  -/
  exact finsum_eq_of_bijective e.symm (Equiv.bijective e.symm) fun x => rfl
  /-
    🎉 no goals
  -/


/-- The summable family given by multiplying every series in a summable family by a scalar. -/
@[simps]
def smulFamily [AddCommMonoid V] [SMulWithZero R V] (f : α → R) (s : SummableFamily Γ V α) :
    SummableFamily Γ V α where
  toFun a := (f a) • s a
  isPWO_iUnion_support' := by
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_4
      α : Type u_5
      β : Type u_6
      inst✝³ : PartialOrder Γ
      inst✝² : AddCommMonoid R
      inst✝¹ : AddCommMonoid V
      inst✝ : SMulWithZero R V
      f : α → R
      s : HahnSeries.SummableFamily Γ V α
      ⊢ (Set.iUnion fun a => ((fun a => HSMul.hSMul (f a) (s a)) a).support).IsPWO
    -/
    refine Set.IsPWO.mono s.isPWO_iUnion_support fun g hg => ?_
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_4
      α : Type u_5
      β : Type u_6
      inst✝³ : PartialOrder Γ
      inst✝² : AddCommMonoid R
      inst✝¹ : AddCommMonoid V
      inst✝ : SMulWithZero R V
      f : α → R
      s : HahnSeries.SummableFamily Γ V α
      g : Γ
      hg : Membership.mem (Set.iUnion fun a => ((fun a => HSMul.hSMul (f a) (s a)) a …
      ⊢ Membership.mem (Set.iUnion fun a => (s a).support) g
    -/
    simp_all only [Set.mem_iUnion, mem_support, smul_coeff, ne_eq]
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_4
      α : Type u_5
      β : Type u_6
      inst✝³ : PartialOrder Γ
      inst✝² : AddCommMonoid R
      inst✝¹ : AddCommMonoid V
      inst✝ : SMulWithZero R V
      f : α → R
      s : HahnSeries.SummableFamily Γ V α
      g : Γ
      hg : Exists fun i => Not (Eq (HSMul.hSMul (f i) ((s i).coeff g)) 0)
      ⊢ Exists fun i => Not (Eq ((s i).coeff g) 0)
    -/
    obtain ⟨i, hi⟩ := hg
    /-
      case intro
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_4
      α : Type u_5
      β : Type u_6
      inst✝³ : PartialOrder Γ
      inst✝² : AddCommMonoid R
      inst✝¹ : AddCommMonoid V
      inst✝ : SMulWithZero R V
      f : α → R
      s : HahnSeries.SummableFamily Γ V α
      g : Γ
      i : α
      hi : Not (Eq (HSMul.hSMul (f i) ((s i).coeff g)) 0)
      ⊢ Exists fun i => Not (Eq ((s i).coeff g) 0)
    -/
    exact Exists.intro i <| right_ne_zero_of_smul hi
    /-
      🎉 no goals
    -/
  finite_co_support' g := by
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_4
      α : Type u_5
      β : Type u_6
      inst✝³ : PartialOrder Γ
      inst✝² : AddCommMonoid R
      inst✝¹ : AddCommMonoid V
      inst✝ : SMulWithZero R V
      f : α → R
      s : HahnSeries.SummableFamily Γ V α
      g : Γ
      ⊢ (setOf fun a => Ne (((fun a => HSMul.hSMul (f a) (s a)) a).coeff g) 0).Finite
    -/
    refine Set.Finite.subset (s.finite_co_support g) fun i hi => ?_
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_4
      α : Type u_5
      β : Type u_6
      inst✝³ : PartialOrder Γ
      inst✝² : AddCommMonoid R
      inst✝¹ : AddCommMonoid V
      inst✝ : SMulWithZero R V
      f : α → R
      s : HahnSeries.SummableFamily Γ V α
      g : Γ
      i : α
      hi : Membership.mem (setOf fun a => Ne (((fun a => HSMul.hSMul (f a) (s a)) a) …
      ⊢ Membership.mem (Function.support fun a => (s a).coeff g) i
    -/
    simp_all only [smul_coeff, ne_eq, Set.mem_setOf_eq, Function.mem_support]
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_4
      α : Type u_5
      β : Type u_6
      inst✝³ : PartialOrder Γ
      inst✝² : AddCommMonoid R
      inst✝¹ : AddCommMonoid V
      inst✝ : SMulWithZero R V
      f : α → R
      s : HahnSeries.SummableFamily Γ V α
      g : Γ
      i : α
      hi : Not (Eq (HSMul.hSMul (f i) ((s i).coeff g)) 0)
      ⊢ Not (Eq ((s i).coeff g) 0)
    -/
    exact right_ne_zero_of_smul hi
    /-
      🎉 no goals
    -/


theorem hsum_smulFamily [AddCommMonoid V] [SMulWithZero R V] (f : α → R)
    (s : SummableFamily Γ V α) (g : Γ) :
    (smulFamily f s).hsum.coeff g = ∑ᶠ i, (f i) • ((s i).coeff g) :=
  rfl


instance : Neg (SummableFamily Γ R α) :=
  ⟨fun s =>
    { toFun := fun a => -s a
      isPWO_iUnion_support' := by
        /-
          Γ : Type u_1
          Γ' : Type u_2
          R : Type u_3
          V : Type u_4
          α : Type u_5
          β : Type u_6
          inst✝¹ : PartialOrder Γ
          inst✝ : AddCommGroup R
          s✝ t : HahnSeries.SummableFamily Γ R α
          a : α
          s : HahnSeries.SummableFamily Γ R α
          ⊢ (Set.iUnion fun a => ((fun a => Neg.neg (s a)) a).support).IsPWO
        -/
        simp_rw [support_neg]
        /-
          Γ : Type u_1
          Γ' : Type u_2
          R : Type u_3
          V : Type u_4
          α : Type u_5
          β : Type u_6
          inst✝¹ : PartialOrder Γ
          inst✝ : AddCommGroup R
          s✝ t : HahnSeries.SummableFamily Γ R α
          a : α
          s : HahnSeries.SummableFamily Γ R α
          ⊢ (Set.iUnion fun a => (s a).support).IsPWO
        -/
        exact s.isPWO_iUnion_support
        /-
          🎉 no goals
        -/
      finite_co_support' := fun g => by
        /-
          Γ : Type u_1
          Γ' : Type u_2
          R : Type u_3
          V : Type u_4
          α : Type u_5
          β : Type u_6
          inst✝¹ : PartialOrder Γ
          inst✝ : AddCommGroup R
          s✝ t : HahnSeries.SummableFamily Γ R α
          a : α
          s : HahnSeries.SummableFamily Γ R α
          g : Γ
          ⊢ (setOf fun a => Ne (((fun a => Neg.neg (s a)) a).coeff g) 0).Finite
        -/
        simp only [neg_coeff', Pi.neg_apply, Ne, neg_eq_zero]
        /-
          Γ : Type u_1
          Γ' : Type u_2
          R : Type u_3
          V : Type u_4
          α : Type u_5
          β : Type u_6
          inst✝¹ : PartialOrder Γ
          inst✝ : AddCommGroup R
          s✝ t : HahnSeries.SummableFamily Γ R α
          a : α
          s : HahnSeries.SummableFamily Γ R α
          g : Γ
          ⊢ (setOf fun a => Not (Eq ((s a).coeff g) 0)).Finite
        -/
        exact s.finite_co_support g }⟩
        /-
          🎉 no goals
        -/


instance : AddCommGroup (SummableFamily Γ R α) :=
  { inferInstanceAs (AddCommMonoid (SummableFamily Γ R α)) with
    zsmul := zsmulRec
    neg_add_cancel := fun a => by
      /-
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        V : Type u_4
        α : Type u_5
        β : Type u_6
        inst✝¹ : PartialOrder Γ
        inst✝ : AddCommGroup R
        s t : HahnSeries.SummableFamily Γ R α
        a✝ : α
        a : HahnSeries.SummableFamily Γ R α
        ⊢ Eq (HAdd.hAdd (Neg.neg a) a) 0
      -/
      ext
      /-
        case h.coeff.h
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        V : Type u_4
        α : Type u_5
        β : Type u_6
        inst✝¹ : PartialOrder Γ
        inst✝ : AddCommGroup R
        s t : HahnSeries.SummableFamily Γ R α
        a✝¹ : α
        a : HahnSeries.SummableFamily Γ R α
        a✝ : α
        x✝ : Γ
        ⊢ Eq (((HAdd.hAdd (Neg.neg a) a) a✝).coeff x✝) ((0 a✝).coeff x✝)
      -/
      apply neg_add_cancel }
      /-
        🎉 no goals
      -/


@[simp]
theorem coe_neg : ⇑(-s) = -s :=
  rfl


theorem neg_apply : (-s) a = -s a :=
  rfl


@[simp]
theorem coe_sub : ⇑(s - t) = s - t :=
  rfl


theorem sub_apply : (s - t) a = s a - t a :=
  rfl


instance [Zero R] [SMulWithZero R V] : SMul R (SummableFamily Γ' V β) :=
  ⟨fun r t =>
    { toFun := r • t
      isPWO_iUnion_support' := t.isPWO_iUnion_support.mono (Set.iUnion_mono fun i =>
        Pi.smul_apply r t i ▸ Function.support_const_smul_subset r _)
      finite_co_support' := by
        /-
          Γ : Type u_1
          Γ' : Type u_2
          R : Type u_3
          V : Type u_4
          α : Type u_5
          β : Type u_6
          inst✝⁴ : PartialOrder Γ
          inst✝³ : PartialOrder Γ'
          inst✝² : AddCommMonoid V
          inst✝¹ : Zero R
          inst✝ : SMulWithZero R V
          r : R
          t : HahnSeries.SummableFamily Γ' V β
          ⊢ ∀ (g : Γ'), (setOf fun a => Ne ((HSMul.hSMul r (⇑t) a).coeff g) 0).Finite
        -/
        intro g
        /-
          Γ : Type u_1
          Γ' : Type u_2
          R : Type u_3
          V : Type u_4
          α : Type u_5
          β : Type u_6
          inst✝⁴ : PartialOrder Γ
          inst✝³ : PartialOrder Γ'
          inst✝² : AddCommMonoid V
          inst✝¹ : Zero R
          inst✝ : SMulWithZero R V
          r : R
          t : HahnSeries.SummableFamily Γ' V β
          g : Γ'
          ⊢ (setOf fun a => Ne ((HSMul.hSMul r (⇑t) a).coeff g) 0).Finite
        -/
        refine (t.finite_co_support g).subset ?_
        /-
          Γ : Type u_1
          Γ' : Type u_2
          R : Type u_3
          V : Type u_4
          α : Type u_5
          β : Type u_6
          inst✝⁴ : PartialOrder Γ
          inst✝³ : PartialOrder Γ'
          inst✝² : AddCommMonoid V
          inst✝¹ : Zero R
          inst✝ : SMulWithZero R V
          r : R
          t : HahnSeries.SummableFamily Γ' V β
          g : Γ'
          ⊢ HasSubset.Subset (setOf fun a => Ne ((HSMul.hSMul r (⇑t) a).coeff g) 0) (Fun …
        -/
        intro i hi
        /-
          Γ : Type u_1
          Γ' : Type u_2
          R : Type u_3
          V : Type u_4
          α : Type u_5
          β : Type u_6
          inst✝⁴ : PartialOrder Γ
          inst✝³ : PartialOrder Γ'
          inst✝² : AddCommMonoid V
          inst✝¹ : Zero R
          inst✝ : SMulWithZero R V
          r : R
          t : HahnSeries.SummableFamily Γ' V β
          g : Γ'
          i : β
          hi : Membership.mem (setOf fun a => Ne ((HSMul.hSMul r (⇑t) a).coeff g) 0) i
          ⊢ Membership.mem (Function.support fun a => (t a).coeff g) i
        -/
        simp only [Pi.smul_apply, smul_coeff, ne_eq, Set.mem_setOf_eq] at hi
        /-
          Γ : Type u_1
          Γ' : Type u_2
          R : Type u_3
          V : Type u_4
          α : Type u_5
          β : Type u_6
          inst✝⁴ : PartialOrder Γ
          inst✝³ : PartialOrder Γ'
          inst✝² : AddCommMonoid V
          inst✝¹ : Zero R
          inst✝ : SMulWithZero R V
          r : R
          t : HahnSeries.SummableFamily Γ' V β
          g : Γ'
          i : β
          hi : Not (Eq (HSMul.hSMul r ((t i).coeff g)) 0)
          ⊢ Membership.mem (Function.support fun a => (t a).coeff g) i
        -/
        simp only [Function.mem_support, ne_eq]
        /-
          Γ : Type u_1
          Γ' : Type u_2
          R : Type u_3
          V : Type u_4
          α : Type u_5
          β : Type u_6
          inst✝⁴ : PartialOrder Γ
          inst✝³ : PartialOrder Γ'
          inst✝² : AddCommMonoid V
          inst✝¹ : Zero R
          inst✝ : SMulWithZero R V
          r : R
          t : HahnSeries.SummableFamily Γ' V β
          g : Γ'
          i : β
          hi : Not (Eq (HSMul.hSMul r ((t i).coeff g)) 0)
          ⊢ Not (Eq ((t i).coeff g) 0)
        -/
        exact right_ne_zero_of_smul hi } ⟩
        /-
          🎉 no goals
        -/


theorem smul_support_subset_prod (s : SummableFamily Γ R α)
    (t : SummableFamily Γ' V β) (gh : Γ × Γ') :
    (Function.support fun (i : α × β) ↦ (s i.1).coeff gh.1 • (t i.2).coeff gh.2) ⊆
    ((s.finite_co_support' gh.1).prod (t.finite_co_support' gh.2)).toFinset := by
  /-
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    V : Type u_4
    α : Type u_5
    β : Type u_6
    inst✝⁴ : PartialOrder Γ
    inst✝³ : PartialOrder Γ'
    inst✝² : AddCommMonoid V
    inst✝¹ : AddCommMonoid R
    inst✝ : SMulWithZero R V
    s : HahnSeries.SummableFamily Γ R α
    t : HahnSeries.SummableFamily Γ' V β
    gh : Prod Γ Γ'
    ⊢ HasSubset.Subset (Function.support fun i => HSMul.hSMul ((s i.1).coeff gh.1) …
  -/
  intro _ hab
  simp_all only [Function.mem_support, ne_eq, Set.Finite.coe_toFinset, Set.mem_prod,
    Set.mem_setOf_eq]
  /-
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    V : Type u_4
    α : Type u_5
    β : Type u_6
    inst✝⁴ : PartialOrder Γ
    inst✝³ : PartialOrder Γ'
    inst✝² : AddCommMonoid V
    inst✝¹ : AddCommMonoid R
    inst✝ : SMulWithZero R V
    s : HahnSeries.SummableFamily Γ R α
    t : HahnSeries.SummableFamily Γ' V β
    gh : Prod Γ Γ'
    a✝ : Prod α β
    hab : Not (Eq (HSMul.hSMul ((s a✝.1).coeff gh.1) ((t a✝.2).coeff gh.2)) 0)
    ⊢ And (Not (Eq ((s.toFun a✝.1).coeff gh.1) 0)) (Not (Eq ((t.toFun a✝.2).coeff  …
  -/
  exact ⟨left_ne_zero_of_smul hab, right_ne_zero_of_smul hab⟩
  /-
    🎉 no goals
  -/


theorem smul_support_finite (s : SummableFamily Γ R α)
    (t : SummableFamily Γ' V β) (gh : Γ × Γ') :
    (Function.support fun (i : α × β) ↦ (s i.1).coeff gh.1 • (t i.2).coeff gh.2).Finite :=
  Set.Finite.subset (Set.toFinite ((s.finite_co_support' gh.1).prod
    (t.finite_co_support' gh.2)).toFinset) (smul_support_subset_prod s t gh)


theorem isPWO_iUnion_support_prod_smul {s : α → HahnSeries Γ R} {t : β → HahnSeries Γ' V}
    (hs : (⋃ a, (s a).support).IsPWO) (ht : (⋃ b, (t b).support).IsPWO) :
    (⋃ (a : α × β), ((fun a ↦ (of R).symm
      ((s a.1) • (of R) (t a.2))) a).support).IsPWO := by
  /-
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    V : Type u_4
    α : Type u_5
    β : Type u_6
    inst✝⁶ : PartialOrder Γ
    inst✝⁵ : PartialOrder Γ'
    inst✝⁴ : AddCommMonoid V
    inst✝³ : AddCommMonoid R
    inst✝² : SMulWithZero R V
    inst✝¹ : VAdd Γ Γ'
    inst✝ : IsOrderedCancelVAdd Γ Γ'
    s : α → HahnSeries Γ R
    t : β → HahnSeries Γ' V
    hs : (Set.iUnion fun a => (s a).support).IsPWO
    ht : (Set.iUnion fun b => (t b).support).IsPWO
    ⊢ (Set.iUnion fun a => ((fun a => (HahnModule.of R).symm (HSMul.hSMul (s a.1)  …
  -/
  apply (hs.vadd ht).mono
  have hsupp : ∀ ab : α × β, support ((fun ab ↦ (of R).symm (s ab.1 • (of R) (t ab.2))) ab) ⊆
      (s ab.1).support +ᵥ (t ab.2).support := by
    intro ab
    refine Set.Subset.trans (fun x hx => ?_) (support_vaddAntidiagonal_subset_vadd
      (hs := (s ab.1).isPWO_support) (ht := (t ab.2).isPWO_support))
    contrapose! hx
    simp only [Set.mem_setOf_eq, not_nonempty_iff_eq_empty] at hx
    rw [mem_support, not_not, HahnModule.smul_coeff, hx, sum_empty]
  /-
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    V : Type u_4
    α : Type u_5
    β : Type u_6
    inst✝⁶ : PartialOrder Γ
    inst✝⁵ : PartialOrder Γ'
    inst✝⁴ : AddCommMonoid V
    inst✝³ : AddCommMonoid R
    inst✝² : SMulWithZero R V
    inst✝¹ : VAdd Γ Γ'
    inst✝ : IsOrderedCancelVAdd Γ Γ'
    s : α → HahnSeries Γ R
    t : β → HahnSeries Γ' V
    hs : (Set.iUnion fun a => (s a).support).IsPWO
    ht : (Set.iUnion fun b => (t b).support).IsPWO
    hsupp : ∀ (ab : Prod α β), HasSubset.Subset ((fun ab => (HahnModule.of R).symm …
    ⊢ HasSubset.Subset (Set.iUnion fun a => ((fun a => (HahnModule.of R).symm (HSM …
  -/
  refine Set.Subset.trans (Set.iUnion_mono fun a => (hsupp a)) ?_
  /-
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    V : Type u_4
    α : Type u_5
    β : Type u_6
    inst✝⁶ : PartialOrder Γ
    inst✝⁵ : PartialOrder Γ'
    inst✝⁴ : AddCommMonoid V
    inst✝³ : AddCommMonoid R
    inst✝² : SMulWithZero R V
    inst✝¹ : VAdd Γ Γ'
    inst✝ : IsOrderedCancelVAdd Γ Γ'
    s : α → HahnSeries Γ R
    t : β → HahnSeries Γ' V
    hs : (Set.iUnion fun a => (s a).support).IsPWO
    ht : (Set.iUnion fun b => (t b).support).IsPWO
    hsupp : ∀ (ab : Prod α β), HasSubset.Subset ((fun ab => (HahnModule.of R).symm …
    ⊢ HasSubset.Subset (Set.iUnion fun i => HVAdd.hVAdd (s i.1).support (t i.2).su …
  -/
  simp_all only [Set.iUnion_subset_iff, Prod.forall]
  exact fun a b => Set.vadd_subset_vadd (Set.subset_iUnion_of_subset a fun x y ↦ y)
    (Set.subset_iUnion_of_subset b fun x y ↦ y)


theorem finite_co_support_prod_smul (s : SummableFamily Γ R α)
    (t : SummableFamily Γ' V β) (g : Γ') :
    Finite {(ab : α × β) |
      ((fun (ab : α × β) ↦ (of R).symm (s ab.1 • (of R) (t ab.2))) ab).coeff g ≠ 0} := by
  apply ((VAddAntidiagonal s.isPWO_iUnion_support t.isPWO_iUnion_support g).finite_toSet.biUnion'
    (fun gh _ => smul_support_finite s t gh)).subset _
  exact fun ab hab => by
    simp only [smul_coeff, ne_eq, Set.mem_setOf_eq] at hab
    obtain ⟨ij, hij⟩ := Finset.exists_ne_zero_of_sum_ne_zero hab
    simp only [mem_coe, mem_vaddAntidiagonal, Set.mem_iUnion, mem_support, ne_eq,
      Function.mem_support, exists_prop, Prod.exists]
    exact ⟨ij.1, ij.2, ⟨⟨ab.1, left_ne_zero_of_smul hij.2⟩, ⟨ab.2, right_ne_zero_of_smul hij.2⟩,
      ((mem_vaddAntidiagonal _ _ _).mp hij.1).2.2⟩, hij.2⟩


/-- An elementwise scalar multiplication of one summable family on another. -/
@[simps]
def smul (s : SummableFamily Γ R α) (t : SummableFamily Γ' V β) :
    (SummableFamily Γ' V (α × β)) where
  toFun ab := (of R).symm (s (ab.1) • ((of R) (t (ab.2))))
  isPWO_iUnion_support' :=
    isPWO_iUnion_support_prod_smul s.isPWO_iUnion_support t.isPWO_iUnion_support
  finite_co_support' g := finite_co_support_prod_smul s t g


@[deprecated (since := "2024-11-17")] noncomputable alias FamilySMul := smul


theorem sum_vAddAntidiagonal_eq (s : SummableFamily Γ R α) (t : SummableFamily Γ' V β) (g : Γ')
    (a : α × β) :
    ∑ x ∈ VAddAntidiagonal (s a.1).isPWO_support' (t a.2).isPWO_support' g, (s a.1).coeff x.1 •
      (t a.2).coeff x.2 = ∑ x ∈ VAddAntidiagonal s.isPWO_iUnion_support' t.isPWO_iUnion_support' g,
      (s a.1).coeff x.1 • (t a.2).coeff x.2 := by
  /-
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    V : Type u_4
    α : Type u_5
    β : Type u_6
    inst✝⁶ : PartialOrder Γ
    inst✝⁵ : PartialOrder Γ'
    inst✝⁴ : AddCommMonoid V
    inst✝³ : AddCommMonoid R
    inst✝² : SMulWithZero R V
    inst✝¹ : VAdd Γ Γ'
    inst✝ : IsOrderedCancelVAdd Γ Γ'
    s : HahnSeries.SummableFamily Γ R α
    t : HahnSeries.SummableFamily Γ' V β
    g : Γ'
    a : Prod α β
    ⊢ Eq ((Finset.VAddAntidiagonal ⋯ ⋯ g).sum fun x => HSMul.hSMul ((s a.1).coeff  …
  -/
  refine sum_subset (fun gh hgh => ?_) fun gh hgh h => ?_
    /-
      case refine_1
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_4
      α : Type u_5
      β : Type u_6
      inst✝⁶ : PartialOrder Γ
      inst✝⁵ : PartialOrder Γ'
      inst✝⁴ : AddCommMonoid V
      inst✝³ : AddCommMonoid R
      inst✝² : SMulWithZero R V
      inst✝¹ : VAdd Γ Γ'
      inst✝ : IsOrderedCancelVAdd Γ Γ'
      s : HahnSeries.SummableFamily Γ R α
      t : HahnSeries.SummableFamily Γ' V β
      g : Γ'
      a : Prod α β
      gh : Prod Γ Γ'
      hgh : Membership.mem (Finset.VAddAntidiagonal ⋯ ⋯ g) gh
      ⊢ Membership.mem (Finset.VAddAntidiagonal ⋯ ⋯ g) gh
    -/
  · simp_all only [mem_vaddAntidiagonal, Function.mem_support, Set.mem_iUnion, mem_support]
    /-
      case refine_1
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_4
      α : Type u_5
      β : Type u_6
      inst✝⁶ : PartialOrder Γ
      inst✝⁵ : PartialOrder Γ'
      inst✝⁴ : AddCommMonoid V
      inst✝³ : AddCommMonoid R
      inst✝² : SMulWithZero R V
      inst✝¹ : VAdd Γ Γ'
      inst✝ : IsOrderedCancelVAdd Γ Γ'
      s : HahnSeries.SummableFamily Γ R α
      t : HahnSeries.SummableFamily Γ' V β
      g : Γ'
      a : Prod α β
      gh : Prod Γ Γ'
      hgh : And (Ne ((s a.1).coeff gh.1) 0) (And (Ne ((t a.2).coeff gh.2) 0) (Eq (HV …
      ⊢ And (Exists fun i => Ne ((s.toFun i).coeff gh.1) 0) (And (Exists fun i => Ne …
    -/
    exact ⟨Exists.intro a.1 hgh.1, Exists.intro a.2 hgh.2.1, trivial⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_4
      α : Type u_5
      β : Type u_6
      inst✝⁶ : PartialOrder Γ
      inst✝⁵ : PartialOrder Γ'
      inst✝⁴ : AddCommMonoid V
      inst✝³ : AddCommMonoid R
      inst✝² : SMulWithZero R V
      inst✝¹ : VAdd Γ Γ'
      inst✝ : IsOrderedCancelVAdd Γ Γ'
      s : HahnSeries.SummableFamily Γ R α
      t : HahnSeries.SummableFamily Γ' V β
      g : Γ'
      a : Prod α β
      gh : Prod Γ Γ'
      hgh : Membership.mem (Finset.VAddAntidiagonal ⋯ ⋯ g) gh
      h : Not (Membership.mem (Finset.VAddAntidiagonal ⋯ ⋯ g) gh)
      ⊢ Eq (HSMul.hSMul ((s a.1).coeff gh.1) ((t a.2).coeff gh.2)) 0
    -/
  · by_cases hs : (s a.1).coeff gh.1 = 0
      /-
        case pos
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        V : Type u_4
        α : Type u_5
        β : Type u_6
        inst✝⁶ : PartialOrder Γ
        inst✝⁵ : PartialOrder Γ'
        inst✝⁴ : AddCommMonoid V
        inst✝³ : AddCommMonoid R
        inst✝² : SMulWithZero R V
        inst✝¹ : VAdd Γ Γ'
        inst✝ : IsOrderedCancelVAdd Γ Γ'
        s : HahnSeries.SummableFamily Γ R α
        t : HahnSeries.SummableFamily Γ' V β
        g : Γ'
        a : Prod α β
        gh : Prod Γ Γ'
        hgh : Membership.mem (Finset.VAddAntidiagonal ⋯ ⋯ g) gh
        h : Not (Membership.mem (Finset.VAddAntidiagonal ⋯ ⋯ g) gh)
        hs : Eq ((s a.1).coeff gh.1) 0
        ⊢ Eq (HSMul.hSMul ((s a.1).coeff gh.1) ((t a.2).coeff gh.2)) 0
      -/
    · exact smul_eq_zero_of_left hs ((t a.2).coeff gh.2)
      /-
        🎉 no goals
      -/
      /-
        case neg
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        V : Type u_4
        α : Type u_5
        β : Type u_6
        inst✝⁶ : PartialOrder Γ
        inst✝⁵ : PartialOrder Γ'
        inst✝⁴ : AddCommMonoid V
        inst✝³ : AddCommMonoid R
        inst✝² : SMulWithZero R V
        inst✝¹ : VAdd Γ Γ'
        inst✝ : IsOrderedCancelVAdd Γ Γ'
        s : HahnSeries.SummableFamily Γ R α
        t : HahnSeries.SummableFamily Γ' V β
        g : Γ'
        a : Prod α β
        gh : Prod Γ Γ'
        hgh : Membership.mem (Finset.VAddAntidiagonal ⋯ ⋯ g) gh
        h : Not (Membership.mem (Finset.VAddAntidiagonal ⋯ ⋯ g) gh)
        hs : Not (Eq ((s a.1).coeff gh.1) 0)
        ⊢ Eq (HSMul.hSMul ((s a.1).coeff gh.1) ((t a.2).coeff gh.2)) 0
      -/
    · simp_all
      /-
        🎉 no goals
      -/


theorem smul_coeff {R} {V} [Semiring R] [AddCommMonoid V] [Module R V]
    (s : SummableFamily Γ R α) (t : SummableFamily Γ' V β) (g : Γ') :
    (smul s t).hsum.coeff g = ∑ gh ∈ VAddAntidiagonal s.isPWO_iUnion_support
      t.isPWO_iUnion_support g, (s.hsum.coeff gh.1) • (t.hsum.coeff gh.2) := by
  /-
    Γ : Type u_1
    Γ' : Type u_2
    α : Type u_5
    β : Type u_6
    inst✝⁶ : PartialOrder Γ
    inst✝⁵ : PartialOrder Γ'
    inst✝⁴ : VAdd Γ Γ'
    inst✝³ : IsOrderedCancelVAdd Γ Γ'
    R : Type u_7
    V : Type u_8
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid V
    inst✝ : Module R V
    s : HahnSeries.SummableFamily Γ R α
    t : HahnSeries.SummableFamily Γ' V β
    g : Γ'
    ⊢ Eq ((s.smul t).hsum.coeff g) ((Finset.VAddAntidiagonal ⋯ ⋯ g).sum fun gh =>  …
  -/
  rw [hsum_coeff]
  /-
    Γ : Type u_1
    Γ' : Type u_2
    α : Type u_5
    β : Type u_6
    inst✝⁶ : PartialOrder Γ
    inst✝⁵ : PartialOrder Γ'
    inst✝⁴ : VAdd Γ Γ'
    inst✝³ : IsOrderedCancelVAdd Γ Γ'
    R : Type u_7
    V : Type u_8
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid V
    inst✝ : Module R V
    s : HahnSeries.SummableFamily Γ R α
    t : HahnSeries.SummableFamily Γ' V β
    g : Γ'
    ⊢ Eq (finsum fun i => ((s.smul t) i).coeff g) ((Finset.VAddAntidiagonal ⋯ ⋯ g) …
  -/
  simp only [hsum_coeff_eq_sum, smul_toFun, HahnModule.smul_coeff, Equiv.symm_apply_apply]
  /-
    Γ : Type u_1
    Γ' : Type u_2
    α : Type u_5
    β : Type u_6
    inst✝⁶ : PartialOrder Γ
    inst✝⁵ : PartialOrder Γ'
    inst✝⁴ : VAdd Γ Γ'
    inst✝³ : IsOrderedCancelVAdd Γ Γ'
    R : Type u_7
    V : Type u_8
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid V
    inst✝ : Module R V
    s : HahnSeries.SummableFamily Γ R α
    t : HahnSeries.SummableFamily Γ' V β
    g : Γ'
    ⊢ Eq (finsum fun i => (Finset.VAddAntidiagonal ⋯ ⋯ g).sum fun x => HSMul.hSMul …
  -/
  simp_rw [sum_vAddAntidiagonal_eq, Finset.smul_sum, Finset.sum_smul]
  /-
    Γ : Type u_1
    Γ' : Type u_2
    α : Type u_5
    β : Type u_6
    inst✝⁶ : PartialOrder Γ
    inst✝⁵ : PartialOrder Γ'
    inst✝⁴ : VAdd Γ Γ'
    inst✝³ : IsOrderedCancelVAdd Γ Γ'
    R : Type u_7
    V : Type u_8
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid V
    inst✝ : Module R V
    s : HahnSeries.SummableFamily Γ R α
    t : HahnSeries.SummableFamily Γ' V β
    g : Γ'
    ⊢ Eq (finsum fun i => (Finset.VAddAntidiagonal ⋯ ⋯ g).sum fun x => HSMul.hSMul …
  -/
  rw [← sum_finsum_comm _ _ <| fun gh _ => smul_support_finite s t gh]
  /-
    Γ : Type u_1
    Γ' : Type u_2
    α : Type u_5
    β : Type u_6
    inst✝⁶ : PartialOrder Γ
    inst✝⁵ : PartialOrder Γ'
    inst✝⁴ : VAdd Γ Γ'
    inst✝³ : IsOrderedCancelVAdd Γ Γ'
    R : Type u_7
    V : Type u_8
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid V
    inst✝ : Module R V
    s : HahnSeries.SummableFamily Γ R α
    t : HahnSeries.SummableFamily Γ' V β
    g : Γ'
    ⊢ Eq ((Finset.VAddAntidiagonal ⋯ ⋯ g).sum fun a => finsum fun b => HSMul.hSMul …
  -/
  refine sum_congr rfl fun gh _ => ?_
  /-
    Γ : Type u_1
    Γ' : Type u_2
    α : Type u_5
    β : Type u_6
    inst✝⁶ : PartialOrder Γ
    inst✝⁵ : PartialOrder Γ'
    inst✝⁴ : VAdd Γ Γ'
    inst✝³ : IsOrderedCancelVAdd Γ Γ'
    R : Type u_7
    V : Type u_8
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid V
    inst✝ : Module R V
    s : HahnSeries.SummableFamily Γ R α
    t : HahnSeries.SummableFamily Γ' V β
    g : Γ'
    gh : Prod Γ Γ'
    x✝ : Membership.mem (Finset.VAddAntidiagonal ⋯ ⋯ g) gh
    ⊢ Eq (finsum fun b => HSMul.hSMul ((s b.1).coeff gh.1) ((t b.2).coeff gh.2)) ( …
  -/
  rw [finsum_eq_sum _ (smul_support_finite s t gh), ← sum_product_right']
  /-
    Γ : Type u_1
    Γ' : Type u_2
    α : Type u_5
    β : Type u_6
    inst✝⁶ : PartialOrder Γ
    inst✝⁵ : PartialOrder Γ'
    inst✝⁴ : VAdd Γ Γ'
    inst✝³ : IsOrderedCancelVAdd Γ Γ'
    R : Type u_7
    V : Type u_8
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid V
    inst✝ : Module R V
    s : HahnSeries.SummableFamily Γ R α
    t : HahnSeries.SummableFamily Γ' V β
    g : Γ'
    gh : Prod Γ Γ'
    x✝ : Membership.mem (Finset.VAddAntidiagonal ⋯ ⋯ g) gh
    ⊢ Eq (⋯.toFinset.sum fun i => HSMul.hSMul ((s i.1).coeff gh.1) ((t i.2).coeff  …
  -/
  refine sum_subset (fun ab hab => ?_) (fun ab _ hab => by simp_all)
  /-
    Γ : Type u_1
    Γ' : Type u_2
    α : Type u_5
    β : Type u_6
    inst✝⁶ : PartialOrder Γ
    inst✝⁵ : PartialOrder Γ'
    inst✝⁴ : VAdd Γ Γ'
    inst✝³ : IsOrderedCancelVAdd Γ Γ'
    R : Type u_7
    V : Type u_8
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid V
    inst✝ : Module R V
    s : HahnSeries.SummableFamily Γ R α
    t : HahnSeries.SummableFamily Γ' V β
    g : Γ'
    gh : Prod Γ Γ'
    x✝ : Membership.mem (Finset.VAddAntidiagonal ⋯ ⋯ g) gh
    ab : Prod α β
    hab : Membership.mem ⋯.toFinset ab
    ⊢ Membership.mem (SProd.sprod (s.coeff gh.1).support (t.coeff gh.2).support) ab
  -/
  have hsupp := smul_support_subset_prod s t gh
  simp_all only [mem_vaddAntidiagonal, Set.mem_iUnion, mem_support, ne_eq, Set.Finite.mem_toFinset,
    Function.mem_support, Set.Finite.coe_toFinset, support_subset_iff, Set.mem_prod,
    Set.mem_setOf_eq, Prod.forall, coeff_support, mem_product]
  /-
    Γ : Type u_1
    Γ' : Type u_2
    α : Type u_5
    β : Type u_6
    inst✝⁶ : PartialOrder Γ
    inst✝⁵ : PartialOrder Γ'
    inst✝⁴ : VAdd Γ Γ'
    inst✝³ : IsOrderedCancelVAdd Γ Γ'
    R : Type u_7
    V : Type u_8
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid V
    inst✝ : Module R V
    s : HahnSeries.SummableFamily Γ R α
    t : HahnSeries.SummableFamily Γ' V β
    g : Γ'
    gh : Prod Γ Γ'
    ab : Prod α β
    x✝ : And (Exists fun i => Not (Eq ((s i).coeff gh.1) 0)) (And (Exists fun i => …
    hab : Not (Eq (HSMul.hSMul ((s ab.1).coeff gh.1) ((t ab.2).coeff gh.2)) 0)
    hsupp : ∀ (a : α) (b : β), Not (Eq (HSMul.hSMul ((s a).coeff gh.1) ((t b).coef …
    ⊢ And (Not (Eq ((s ab.1).coeff gh.1) 0)) (Not (Eq ((t ab.2).coeff gh.2) 0))
  -/
  exact hsupp ab.1 ab.2 hab
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-17")] alias family_smul_coeff := smul_coeff


theorem smul_hsum {R} {V} [Semiring R] [AddCommMonoid V] [Module R V]
    (s : SummableFamily Γ R α) (t : SummableFamily Γ' V β) :
    (smul s t).hsum = (of R).symm (s.hsum • (of R) (t.hsum)) := by
  /-
    Γ : Type u_1
    Γ' : Type u_2
    α : Type u_5
    β : Type u_6
    inst✝⁶ : PartialOrder Γ
    inst✝⁵ : PartialOrder Γ'
    inst✝⁴ : VAdd Γ Γ'
    inst✝³ : IsOrderedCancelVAdd Γ Γ'
    R : Type u_7
    V : Type u_8
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid V
    inst✝ : Module R V
    s : HahnSeries.SummableFamily Γ R α
    t : HahnSeries.SummableFamily Γ' V β
    ⊢ Eq (s.smul t).hsum ((HahnModule.of R).symm (HSMul.hSMul s.hsum ((HahnModule. …
  -/
  ext g
  /-
    case coeff.h
    Γ : Type u_1
    Γ' : Type u_2
    α : Type u_5
    β : Type u_6
    inst✝⁶ : PartialOrder Γ
    inst✝⁵ : PartialOrder Γ'
    inst✝⁴ : VAdd Γ Γ'
    inst✝³ : IsOrderedCancelVAdd Γ Γ'
    R : Type u_7
    V : Type u_8
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid V
    inst✝ : Module R V
    s : HahnSeries.SummableFamily Γ R α
    t : HahnSeries.SummableFamily Γ' V β
    g : Γ'
    ⊢ Eq ((s.smul t).hsum.coeff g) (((HahnModule.of R).symm (HSMul.hSMul s.hsum (( …
  -/
  rw [smul_coeff s t g, HahnModule.smul_coeff, Equiv.symm_apply_apply]
  refine Eq.symm (sum_of_injOn (fun a ↦ a) (fun _ _ _ _ h ↦ h) (fun _ hgh => ?_)
    (fun gh _ hgh => ?_) fun _ _ => by simp)
    /-
      case coeff.h.refine_1
      Γ : Type u_1
      Γ' : Type u_2
      α : Type u_5
      β : Type u_6
      inst✝⁶ : PartialOrder Γ
      inst✝⁵ : PartialOrder Γ'
      inst✝⁴ : VAdd Γ Γ'
      inst✝³ : IsOrderedCancelVAdd Γ Γ'
      R : Type u_7
      V : Type u_8
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid V
      inst✝ : Module R V
      s : HahnSeries.SummableFamily Γ R α
      t : HahnSeries.SummableFamily Γ' V β
      g : Γ'
      x✝ : Prod Γ Γ'
      hgh : Membership.mem (↑(Finset.VAddAntidiagonal ⋯ ⋯ g)) x✝
      ⊢ Membership.mem (↑(Finset.VAddAntidiagonal ⋯ ⋯ g)) ((fun a => a) x✝)
    -/
  · simp_all only [mem_coe, mem_vaddAntidiagonal, mem_support, ne_eq, Set.mem_iUnion, and_true]
    /-
      case coeff.h.refine_1
      Γ : Type u_1
      Γ' : Type u_2
      α : Type u_5
      β : Type u_6
      inst✝⁶ : PartialOrder Γ
      inst✝⁵ : PartialOrder Γ'
      inst✝⁴ : VAdd Γ Γ'
      inst✝³ : IsOrderedCancelVAdd Γ Γ'
      R : Type u_7
      V : Type u_8
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid V
      inst✝ : Module R V
      s : HahnSeries.SummableFamily Γ R α
      t : HahnSeries.SummableFamily Γ' V β
      g : Γ'
      x✝ : Prod Γ Γ'
      hgh : And (Not (Eq (s.hsum.coeff x✝.1) 0)) (And (Not (Eq (t.hsum.coeff x✝.2) 0 …
      ⊢ And (Exists fun i => Not (Eq ((s i).coeff x✝.1) 0)) (Exists fun i => Not (Eq …
    -/
    constructor
      /-
        case coeff.h.refine_1.left
        Γ : Type u_1
        Γ' : Type u_2
        α : Type u_5
        β : Type u_6
        inst✝⁶ : PartialOrder Γ
        inst✝⁵ : PartialOrder Γ'
        inst✝⁴ : VAdd Γ Γ'
        inst✝³ : IsOrderedCancelVAdd Γ Γ'
        R : Type u_7
        V : Type u_8
        inst✝² : Semiring R
        inst✝¹ : AddCommMonoid V
        inst✝ : Module R V
        s : HahnSeries.SummableFamily Γ R α
        t : HahnSeries.SummableFamily Γ' V β
        g : Γ'
        x✝ : Prod Γ Γ'
        hgh : And (Not (Eq (s.hsum.coeff x✝.1) 0)) (And (Not (Eq (t.hsum.coeff x✝.2) 0 …
        ⊢ Exists fun i => Not (Eq ((s i).coeff x✝.1) 0)
      -/
    · rw [hsum_coeff_eq_sum] at hgh
      /-
        case coeff.h.refine_1.left
        Γ : Type u_1
        Γ' : Type u_2
        α : Type u_5
        β : Type u_6
        inst✝⁶ : PartialOrder Γ
        inst✝⁵ : PartialOrder Γ'
        inst✝⁴ : VAdd Γ Γ'
        inst✝³ : IsOrderedCancelVAdd Γ Γ'
        R : Type u_7
        V : Type u_8
        inst✝² : Semiring R
        inst✝¹ : AddCommMonoid V
        inst✝ : Module R V
        s : HahnSeries.SummableFamily Γ R α
        t : HahnSeries.SummableFamily Γ' V β
        g : Γ'
        x✝ : Prod Γ Γ'
        hgh : And (Not (Eq ((s.coeff x✝.1).support.sum fun i => (s i).coeff x✝.1) 0))  …
        ⊢ Exists fun i => Not (Eq ((s i).coeff x✝.1) 0)
      -/
      have h' := Finset.exists_ne_zero_of_sum_ne_zero hgh.1
      /-
        case coeff.h.refine_1.left
        Γ : Type u_1
        Γ' : Type u_2
        α : Type u_5
        β : Type u_6
        inst✝⁶ : PartialOrder Γ
        inst✝⁵ : PartialOrder Γ'
        inst✝⁴ : VAdd Γ Γ'
        inst✝³ : IsOrderedCancelVAdd Γ Γ'
        R : Type u_7
        V : Type u_8
        inst✝² : Semiring R
        inst✝¹ : AddCommMonoid V
        inst✝ : Module R V
        s : HahnSeries.SummableFamily Γ R α
        t : HahnSeries.SummableFamily Γ' V β
        g : Γ'
        x✝ : Prod Γ Γ'
        hgh : And (Not (Eq ((s.coeff x✝.1).support.sum fun i => (s i).coeff x✝.1) 0))  …
        h' : Exists fun a => And (Membership.mem (s.coeff x✝.1).support a) (Ne ((s a). …
        ⊢ Exists fun i => Not (Eq ((s i).coeff x✝.1) 0)
      -/
      simpa using h'
      /-
        🎉 no goals
      -/
      /-
        case coeff.h.refine_1.right
        Γ : Type u_1
        Γ' : Type u_2
        α : Type u_5
        β : Type u_6
        inst✝⁶ : PartialOrder Γ
        inst✝⁵ : PartialOrder Γ'
        inst✝⁴ : VAdd Γ Γ'
        inst✝³ : IsOrderedCancelVAdd Γ Γ'
        R : Type u_7
        V : Type u_8
        inst✝² : Semiring R
        inst✝¹ : AddCommMonoid V
        inst✝ : Module R V
        s : HahnSeries.SummableFamily Γ R α
        t : HahnSeries.SummableFamily Γ' V β
        g : Γ'
        x✝ : Prod Γ Γ'
        hgh : And (Not (Eq (s.hsum.coeff x✝.1) 0)) (And (Not (Eq (t.hsum.coeff x✝.2) 0 …
        ⊢ Exists fun i => Not (Eq ((t i).coeff x✝.2) 0)
      -/
    · by_contra hi
      /-
        case coeff.h.refine_1.right
        Γ : Type u_1
        Γ' : Type u_2
        α : Type u_5
        β : Type u_6
        inst✝⁶ : PartialOrder Γ
        inst✝⁵ : PartialOrder Γ'
        inst✝⁴ : VAdd Γ Γ'
        inst✝³ : IsOrderedCancelVAdd Γ Γ'
        R : Type u_7
        V : Type u_8
        inst✝² : Semiring R
        inst✝¹ : AddCommMonoid V
        inst✝ : Module R V
        s : HahnSeries.SummableFamily Γ R α
        t : HahnSeries.SummableFamily Γ' V β
        g : Γ'
        x✝ : Prod Γ Γ'
        hgh : And (Not (Eq (s.hsum.coeff x✝.1) 0)) (And (Not (Eq (t.hsum.coeff x✝.2) 0 …
        hi : Not (Exists fun i => Not (Eq ((t i).coeff x✝.2) 0))
        ⊢ False
      -/
      simp_all
      /-
        🎉 no goals
      -/
    /-
      case coeff.h.refine_2
      Γ : Type u_1
      Γ' : Type u_2
      α : Type u_5
      β : Type u_6
      inst✝⁶ : PartialOrder Γ
      inst✝⁵ : PartialOrder Γ'
      inst✝⁴ : VAdd Γ Γ'
      inst✝³ : IsOrderedCancelVAdd Γ Γ'
      R : Type u_7
      V : Type u_8
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid V
      inst✝ : Module R V
      s : HahnSeries.SummableFamily Γ R α
      t : HahnSeries.SummableFamily Γ' V β
      g : Γ'
      gh : Prod Γ Γ'
      x✝ : Membership.mem (Finset.VAddAntidiagonal ⋯ ⋯ g) gh
      hgh : Not (Membership.mem (Set.image (fun a => a) ↑(Finset.VAddAntidiagonal ⋯  …
      ⊢ Eq (HSMul.hSMul (s.hsum.coeff gh.1) (t.hsum.coeff gh.2)) 0
    -/
  · simp only [Set.image_id', mem_coe, mem_vaddAntidiagonal, mem_support, ne_eq, not_and] at hgh
    /-
      case coeff.h.refine_2
      Γ : Type u_1
      Γ' : Type u_2
      α : Type u_5
      β : Type u_6
      inst✝⁶ : PartialOrder Γ
      inst✝⁵ : PartialOrder Γ'
      inst✝⁴ : VAdd Γ Γ'
      inst✝³ : IsOrderedCancelVAdd Γ Γ'
      R : Type u_7
      V : Type u_8
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid V
      inst✝ : Module R V
      s : HahnSeries.SummableFamily Γ R α
      t : HahnSeries.SummableFamily Γ' V β
      g : Γ'
      gh : Prod Γ Γ'
      x✝ : Membership.mem (Finset.VAddAntidiagonal ⋯ ⋯ g) gh
      hgh : Not (Eq (s.hsum.coeff gh.1) 0) → Not (Eq (t.hsum.coeff gh.2) 0) → Not (E …
      ⊢ Eq (HSMul.hSMul (s.hsum.coeff gh.1) (t.hsum.coeff gh.2)) 0
    -/
    by_cases h : s.hsum.coeff gh.1 = 0
      /-
        case pos
        Γ : Type u_1
        Γ' : Type u_2
        α : Type u_5
        β : Type u_6
        inst✝⁶ : PartialOrder Γ
        inst✝⁵ : PartialOrder Γ'
        inst✝⁴ : VAdd Γ Γ'
        inst✝³ : IsOrderedCancelVAdd Γ Γ'
        R : Type u_7
        V : Type u_8
        inst✝² : Semiring R
        inst✝¹ : AddCommMonoid V
        inst✝ : Module R V
        s : HahnSeries.SummableFamily Γ R α
        t : HahnSeries.SummableFamily Γ' V β
        g : Γ'
        gh : Prod Γ Γ'
        x✝ : Membership.mem (Finset.VAddAntidiagonal ⋯ ⋯ g) gh
        hgh : Not (Eq (s.hsum.coeff gh.1) 0) → Not (Eq (t.hsum.coeff gh.2) 0) → Not (E …
        h : Eq (s.hsum.coeff gh.1) 0
        ⊢ Eq (HSMul.hSMul (s.hsum.coeff gh.1) (t.hsum.coeff gh.2)) 0
      -/
    · exact smul_eq_zero_of_left h (t.hsum.coeff gh.2)
      /-
        🎉 no goals
      -/
      /-
        case neg
        Γ : Type u_1
        Γ' : Type u_2
        α : Type u_5
        β : Type u_6
        inst✝⁶ : PartialOrder Γ
        inst✝⁵ : PartialOrder Γ'
        inst✝⁴ : VAdd Γ Γ'
        inst✝³ : IsOrderedCancelVAdd Γ Γ'
        R : Type u_7
        V : Type u_8
        inst✝² : Semiring R
        inst✝¹ : AddCommMonoid V
        inst✝ : Module R V
        s : HahnSeries.SummableFamily Γ R α
        t : HahnSeries.SummableFamily Γ' V β
        g : Γ'
        gh : Prod Γ Γ'
        x✝ : Membership.mem (Finset.VAddAntidiagonal ⋯ ⋯ g) gh
        hgh : Not (Eq (s.hsum.coeff gh.1) 0) → Not (Eq (t.hsum.coeff gh.2) 0) → Not (E …
        h : Not (Eq (s.hsum.coeff gh.1) 0)
        ⊢ Eq (HSMul.hSMul (s.hsum.coeff gh.1) (t.hsum.coeff gh.2)) 0
      -/
    · simp_all
      /-
        🎉 no goals
      -/


@[deprecated (since := "2024-11-17")] alias hsum_family_smul := smul_hsum


instance [AddCommMonoid R] [SMulWithZero R V] : SMul (HahnSeries Γ R) (SummableFamily Γ' V β) where
  smul x t := Equiv (Equiv.punitProd β) <| smul (single x) t


theorem smul_eq {x : HahnSeries Γ R} {t : SummableFamily Γ' V β} :
    x • t = Equiv (Equiv.punitProd β) (smul (single x) t) :=
  rfl


@[simp]
theorem smul_apply {x : HahnSeries Γ R} {s : SummableFamily Γ' V α} {a : α} :
    (x • s) a = (of R).symm (x • of R (s a)) :=
  rfl


@[simp]
theorem hsum_smul_module {R} {V} [Semiring R] [AddCommMonoid V] [Module R V] {x : HahnSeries Γ R}
    {s : SummableFamily Γ' V α} :
    (x • s).hsum = (of R).symm (x • of R s.hsum) := by
  /-
    Γ : Type u_1
    Γ' : Type u_2
    α : Type u_5
    inst✝⁶ : PartialOrder Γ
    inst✝⁵ : PartialOrder Γ'
    inst✝⁴ : VAdd Γ Γ'
    inst✝³ : IsOrderedCancelVAdd Γ Γ'
    R : Type u_7
    V : Type u_8
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid V
    inst✝ : Module R V
    x : HahnSeries Γ R
    s : HahnSeries.SummableFamily Γ' V α
    ⊢ Eq (HSMul.hSMul x s).hsum ((HahnModule.of R).symm (HSMul.hSMul x ((HahnModul …
  -/
  rw [smul_eq, hsum_equiv, smul_hsum, hsum_single]
  /-
    🎉 no goals
  -/


instance [AddCommMonoid V] [Module R V] : Module (HahnSeries Γ R) (SummableFamily Γ' V α) where
  smul := (· • ·)
                                 /-
                                   Γ : Type u_1
                                   Γ' : Type u_2
                                   R : Type u_3
                                   V : Type u_4
                                   α : Type u_5
                                   β : Type u_6
                                   inst✝⁶ : OrderedCancelAddCommMonoid Γ
                                   inst✝⁵ : PartialOrder Γ'
                                   inst✝⁴ : AddAction Γ Γ'
                                   inst✝³ : IsOrderedCancelVAdd Γ Γ'
                                   inst✝² : Semiring R
                                   inst✝¹ : AddCommMonoid V
                                   inst✝ : Module R V
                                   x✝¹ : HahnSeries Γ R
                                   x✝ : α
                                   ⊢ Eq ((HSMul.hSMul x✝¹ 0) x✝) (0 x✝)
                                 -/
  smul_zero _ := ext fun _ => by simp
                                /-
                                  Γ : Type u_1
                                  Γ' : Type u_2
                                  R : Type u_3
                                  V : Type u_4
                                  α : Type u_5
                                  β : Type u_6
                                  inst✝⁶ : OrderedCancelAddCommMonoid Γ
                                  inst✝⁵ : PartialOrder Γ'
                                  inst✝⁴ : AddAction Γ Γ'
                                  inst✝³ : IsOrderedCancelVAdd Γ Γ'
                                  inst✝² : Semiring R
                                  inst✝¹ : AddCommMonoid V
                                  inst✝ : Module R V
                                  x✝¹ : HahnSeries.SummableFamily Γ' V α
                                  x✝ : α
                                  ⊢ Eq ((HSMul.hSMul 1 x✝¹) x✝) (x✝¹ x✝)
                                -/
                                 /-
                                   🎉 no goals
                                 -/
                                /-
                                  🎉 no goals
                                -/
                                 /-
                                   Γ : Type u_1
                                   Γ' : Type u_2
                                   R : Type u_3
                                   V : Type u_4
                                   α : Type u_5
                                   β : Type u_6
                                   inst✝⁶ : OrderedCancelAddCommMonoid Γ
                                   inst✝⁵ : PartialOrder Γ'
                                   inst✝⁴ : AddAction Γ Γ'
                                   inst✝³ : IsOrderedCancelVAdd Γ Γ'
                                   inst✝² : Semiring R
                                   inst✝¹ : AddCommMonoid V
                                   inst✝ : Module R V
                                   x✝¹ : HahnSeries.SummableFamily Γ' V α
                                   x✝ : α
                                   ⊢ Eq ((HSMul.hSMul 0 x✝¹) x✝) (0 x✝)
                                 -/
  zero_smul _ := ext fun _ => by simp
                                    /-
                                      Γ : Type u_1
                                      Γ' : Type u_2
                                      R : Type u_3
                                      V : Type u_4
                                      α : Type u_5
                                      β : Type u_6
                                      inst✝⁶ : OrderedCancelAddCommMonoid Γ
                                      inst✝⁵ : PartialOrder Γ'
                                      inst✝⁴ : AddAction Γ Γ'
                                      inst✝³ : IsOrderedCancelVAdd Γ Γ'
                                      inst✝² : Semiring R
                                      inst✝¹ : AddCommMonoid V
                                      inst✝ : Module R V
                                      x✝³ x✝² : HahnSeries Γ R
                                      x✝¹ : HahnSeries.SummableFamily Γ' V α
                                      x✝ : α
                                      ⊢ Eq ((HSMul.hSMul (HMul.hMul x✝³ x✝²) x✝¹) x✝) ((HSMul.hSMul x✝³ (HSMul.hSMul …
                                    -/
                                     /-
                                       Γ : Type u_1
                                       Γ' : Type u_2
                                       R : Type u_3
                                       V : Type u_4
                                       α : Type u_5
                                       β : Type u_6
                                       inst✝⁶ : OrderedCancelAddCommMonoid Γ
                                       inst✝⁵ : PartialOrder Γ'
                                       inst✝⁴ : AddAction Γ Γ'
                                       inst✝³ : IsOrderedCancelVAdd Γ Γ'
                                       inst✝² : Semiring R
                                       inst✝¹ : AddCommMonoid V
                                       inst✝ : Module R V
                                       x✝³ x✝² : HahnSeries Γ R
                                       x✝¹ : HahnSeries.SummableFamily Γ' V α
                                       x✝ : α
                                       ⊢ Eq ((HSMul.hSMul (HAdd.hAdd x✝³ x✝²) x✝¹) x✝) ((HAdd.hAdd (HSMul.hSMul x✝³ x …
                                     -/
                                    /-
                                      🎉 no goals
                                    -/
                                    /-
                                      Γ : Type u_1
                                      Γ' : Type u_2
                                      R : Type u_3
                                      V : Type u_4
                                      α : Type u_5
                                      β : Type u_6
                                      inst✝⁶ : OrderedCancelAddCommMonoid Γ
                                      inst✝⁵ : PartialOrder Γ'
                                      inst✝⁴ : AddAction Γ Γ'
                                      inst✝³ : IsOrderedCancelVAdd Γ Γ'
                                      inst✝² : Semiring R
                                      inst✝¹ : AddCommMonoid V
                                      inst✝ : Module R V
                                      x✝³ : HahnSeries Γ R
                                      x✝² x✝¹ : HahnSeries.SummableFamily Γ' V α
                                      x✝ : α
                                      ⊢ Eq ((HSMul.hSMul x✝³ (HAdd.hAdd x✝² x✝¹)) x✝) ((HAdd.hAdd (HSMul.hSMul x✝³ x …
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
  one_smul _ := ext fun _ => by rw [smul_apply, HahnModule.one_smul', Equiv.symm_apply_apply]
  add_smul _ _ _  := ext fun _ => by simp [add_smul]
  smul_add _ _ _ := ext fun _ => by simp
  mul_smul _ _ _ := ext fun _ => by simp [HahnModule.instModule.mul_smul]


theorem hsum_smul {x : HahnSeries Γ R} {s : SummableFamily Γ R α} :
    (x • s).hsum = x * s.hsum := by
  /-
    Γ : Type u_1
    R : Type u_3
    α : Type u_5
    inst✝¹ : OrderedCancelAddCommMonoid Γ
    inst✝ : Semiring R
    x : HahnSeries Γ R
    s : HahnSeries.SummableFamily Γ R α
    ⊢ Eq (HSMul.hSMul x s).hsum (HMul.hMul x s.hsum)
  -/
  rw [hsum_smul_module, of_symm_smul_of_eq_mul]
  /-
    🎉 no goals
  -/


/-- The summation of a `summable_family` as a `LinearMap`. -/
@[simps]
def lsum : SummableFamily Γ R α →ₗ[HahnSeries Γ R] HahnSeries Γ R where
  toFun := hsum
  map_add' _ _ := hsum_add
  map_smul' _ _ := hsum_smul


@[simp]
theorem hsum_sub {R : Type*} [Ring R] {s t : SummableFamily Γ R α} :
    (s - t).hsum = s.hsum - t.hsum := by
  /-
    Γ : Type u_1
    α : Type u_5
    inst✝¹ : OrderedCancelAddCommMonoid Γ
    R : Type u_7
    inst✝ : Ring R
    s t : HahnSeries.SummableFamily Γ R α
    ⊢ Eq (HSub.hSub s t).hsum (HSub.hSub s.hsum t.hsum)
  -/
  rw [← lsum_apply, LinearMap.map_sub, lsum_apply, lsum_apply]
  /-
    🎉 no goals
  -/


theorem isPWO_iUnion_support_prod_mul {s : α → HahnSeries Γ R} {t : β → HahnSeries Γ R}
    (hs : (⋃ a, (s a).support).IsPWO) (ht : (⋃ b, (t b).support).IsPWO) :
    (⋃ (a : α × β), ((fun a ↦ ((s a.1) * (t a.2))) a).support).IsPWO :=
  isPWO_iUnion_support_prod_smul hs ht


theorem finite_co_support_prod_mul (s : SummableFamily Γ R α)
    (t : SummableFamily Γ R β) (g : Γ) :
    Finite {(a : α × β) | ((fun (a : α × β) ↦ (s a.1 * t a.2)) a).coeff g ≠ 0} :=
  finite_co_support_prod_smul s t g


/-- A summable family given by pointwise multiplication of a pair of summable families. -/
@[simps]
def mul (s : SummableFamily Γ R α) (t : SummableFamily Γ R β) :
    (SummableFamily Γ R (α × β)) where
  toFun a := s (a.1) * t (a.2)
  isPWO_iUnion_support' :=
    isPWO_iUnion_support_prod_mul s.isPWO_iUnion_support t.isPWO_iUnion_support
  finite_co_support' g := finite_co_support_prod_mul s t g


theorem mul_eq_smul {β : Type*} (s : SummableFamily Γ R α) (t : SummableFamily Γ R β) :
    mul s t = smul s t :=
  rfl


theorem mul_coeff {β : Type*} (s : SummableFamily Γ R α) (t : SummableFamily Γ R β) (g : Γ) :
    (mul s t).hsum.coeff g = ∑ gh ∈ addAntidiagonal s.isPWO_iUnion_support
      t.isPWO_iUnion_support g, (s.hsum.coeff gh.1) * (t.hsum.coeff gh.2) := by
  /-
    Γ : Type u_1
    R : Type u_3
    α : Type u_5
    inst✝¹ : OrderedCancelAddCommMonoid Γ
    inst✝ : Semiring R
    β : Type u_7
    s : HahnSeries.SummableFamily Γ R α
    t : HahnSeries.SummableFamily Γ R β
    g : Γ
    ⊢ Eq ((s.mul t).hsum.coeff g) ((Finset.addAntidiagonal ⋯ ⋯ g).sum fun gh => HM …
  -/
  simp_rw [← smul_eq_mul, mul_eq_smul]
  /-
    Γ : Type u_1
    R : Type u_3
    α : Type u_5
    inst✝¹ : OrderedCancelAddCommMonoid Γ
    inst✝ : Semiring R
    β : Type u_7
    s : HahnSeries.SummableFamily Γ R α
    t : HahnSeries.SummableFamily Γ R β
    g : Γ
    ⊢ Eq ((s.smul t).hsum.coeff g) ((Finset.addAntidiagonal ⋯ ⋯ g).sum fun x => HS …
  -/
  exact smul_coeff s t g
  /-
    🎉 no goals
  -/


theorem hsum_mul {β : Type*} (s : SummableFamily Γ R α) (t : SummableFamily Γ R β) :
    (mul s t).hsum = s.hsum * t.hsum := by
  /-
    Γ : Type u_1
    R : Type u_3
    α : Type u_5
    inst✝¹ : OrderedCancelAddCommMonoid Γ
    inst✝ : Semiring R
    β : Type u_7
    s : HahnSeries.SummableFamily Γ R α
    t : HahnSeries.SummableFamily Γ R β
    ⊢ Eq (s.mul t).hsum (HMul.hMul s.hsum t.hsum)
  -/
  rw [← smul_eq_mul, mul_eq_smul]
  /-
    Γ : Type u_1
    R : Type u_3
    α : Type u_5
    inst✝¹ : OrderedCancelAddCommMonoid Γ
    inst✝ : Semiring R
    β : Type u_7
    s : HahnSeries.SummableFamily Γ R α
    t : HahnSeries.SummableFamily Γ R β
    ⊢ Eq (s.smul t).hsum (HSMul.hSMul s.hsum t.hsum)
  -/
  exact smul_hsum s t
  /-
    🎉 no goals
  -/


/-- A family with only finitely many nonzero elements is summable. -/
def ofFinsupp (f : α →₀ HahnSeries Γ R) : SummableFamily Γ R α where
  toFun := f
  isPWO_iUnion_support' := by
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_4
      α✝ : Type u_5
      β : Type u_6
      inst✝¹ : PartialOrder Γ
      inst✝ : AddCommMonoid R
      α : Type u_7
      f : Finsupp α (HahnSeries Γ R)
      ⊢ (Set.iUnion fun a => (f a).support).IsPWO
    -/
    apply (f.support.isPWO_bUnion.2 fun a _ => (f a).isPWO_support).mono
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_4
      α✝ : Type u_5
      β : Type u_6
      inst✝¹ : PartialOrder Γ
      inst✝ : AddCommMonoid R
      α : Type u_7
      f : Finsupp α (HahnSeries Γ R)
      ⊢ HasSubset.Subset (Set.iUnion fun a => (f a).support) (Set.iUnion fun i => Se …
    -/
    refine Set.iUnion_subset_iff.2 fun a g hg => ?_
    have haf : a ∈ f.support := by
      rw [Finsupp.mem_support_iff, ← support_nonempty_iff]
      exact ⟨g, hg⟩
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_4
      α✝ : Type u_5
      β : Type u_6
      inst✝¹ : PartialOrder Γ
      inst✝ : AddCommMonoid R
      α : Type u_7
      f : Finsupp α (HahnSeries Γ R)
      a : α
      g : Γ
      hg : Membership.mem (f a).support g
      haf : Membership.mem f.support a
      ⊢ Membership.mem (Set.iUnion fun i => Set.iUnion fun h => (f i).support) g
    -/
    exact Set.mem_biUnion haf hg
    /-
      🎉 no goals
    -/
  finite_co_support' g := by
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_4
      α✝ : Type u_5
      β : Type u_6
      inst✝¹ : PartialOrder Γ
      inst✝ : AddCommMonoid R
      α : Type u_7
      f : Finsupp α (HahnSeries Γ R)
      g : Γ
      ⊢ (setOf fun a => Ne ((f a).coeff g) 0).Finite
    -/
    refine f.support.finite_toSet.subset fun a ha => ?_
    simp only [coeff.addMonoidHom_apply, mem_coe, Finsupp.mem_support_iff, Ne,
      Function.mem_support]
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_4
      α✝ : Type u_5
      β : Type u_6
      inst✝¹ : PartialOrder Γ
      inst✝ : AddCommMonoid R
      α : Type u_7
      f : Finsupp α (HahnSeries Γ R)
      g : Γ
      a : α
      ha : Membership.mem (setOf fun a => Ne ((f a).coeff g) 0) a
      ⊢ Not (Eq (f a) 0)
    -/
    contrapose! ha
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_4
      α✝ : Type u_5
      β : Type u_6
      inst✝¹ : PartialOrder Γ
      inst✝ : AddCommMonoid R
      α : Type u_7
      f : Finsupp α (HahnSeries Γ R)
      g : Γ
      a : α
      ha : Eq (f a) 0
      ⊢ Not (Membership.mem (setOf fun a => Ne ((f a).coeff g) 0) a)
    -/
    simp [ha]
    /-
      🎉 no goals
    -/


@[simp]
theorem coe_ofFinsupp {f : α →₀ HahnSeries Γ R} : ⇑(SummableFamily.ofFinsupp f) = f :=
  rfl


@[simp]
theorem hsum_ofFinsupp {f : α →₀ HahnSeries Γ R} : (ofFinsupp f).hsum = f.sum fun _ => id := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : AddCommMonoid R
    α : Type u_7
    f : Finsupp α (HahnSeries Γ R)
    ⊢ Eq (HahnSeries.SummableFamily.ofFinsupp f).hsum (f.sum fun x => id)
  -/
  ext g
  /-
    case coeff.h
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : AddCommMonoid R
    α : Type u_7
    f : Finsupp α (HahnSeries Γ R)
    g : Γ
    ⊢ Eq ((HahnSeries.SummableFamily.ofFinsupp f).hsum.coeff g) ((f.sum fun x => i …
  -/
  simp only [hsum_coeff, coe_ofFinsupp, Finsupp.sum, Ne]
  /-
    case coeff.h
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : AddCommMonoid R
    α : Type u_7
    f : Finsupp α (HahnSeries Γ R)
    g : Γ
    ⊢ Eq (finsum fun i => (f i).coeff g) ((f.support.sum fun x => id (f x)).coeff g)
  -/
  simp_rw [← coeff.addMonoidHom_apply, id]
  /-
    case coeff.h
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : AddCommMonoid R
    α : Type u_7
    f : Finsupp α (HahnSeries Γ R)
    g : Γ
    ⊢ Eq (finsum fun i => (HahnSeries.coeff.addMonoidHom g) (f i)) ((HahnSeries.co …
  -/
  rw [map_sum, finsum_eq_sum_of_support_subset]
  /-
    case coeff.h.h
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : AddCommMonoid R
    α : Type u_7
    f : Finsupp α (HahnSeries Γ R)
    g : Γ
    ⊢ HasSubset.Subset (Function.support fun i => (HahnSeries.coeff.addMonoidHom g …
  -/
  intro x h
  /-
    case coeff.h.h
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : AddCommMonoid R
    α : Type u_7
    f : Finsupp α (HahnSeries Γ R)
    g : Γ
    x : α
    h : Membership.mem (Function.support fun i => (HahnSeries.coeff.addMonoidHom g …
    ⊢ Membership.mem (↑f.support) x
  -/
  simp only [coeff.addMonoidHom_apply, mem_coe, Finsupp.mem_support_iff, Ne]
  /-
    case coeff.h.h
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : AddCommMonoid R
    α : Type u_7
    f : Finsupp α (HahnSeries Γ R)
    g : Γ
    x : α
    h : Membership.mem (Function.support fun i => (HahnSeries.coeff.addMonoidHom g …
    ⊢ Not (Eq (f x) 0)
  -/
  contrapose! h
  /-
    case coeff.h.h
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : AddCommMonoid R
    α : Type u_7
    f : Finsupp α (HahnSeries Γ R)
    g : Γ
    x : α
    h : Eq (f x) 0
    ⊢ Not (Membership.mem (Function.support fun i => (HahnSeries.coeff.addMonoidHo …
  -/
  simp [h]
  /-
    🎉 no goals
  -/


open Classical in
/-- A summable family can be reindexed by an embedding without changing its sum. -/
def embDomain (s : SummableFamily Γ R α) (f : α ↪ β) : SummableFamily Γ R β where
  toFun b := if h : b ∈ Set.range f then s (Classical.choose h) else 0
  isPWO_iUnion_support' := by
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_4
      α✝ : Type u_5
      β✝ : Type u_6
      inst✝¹ : PartialOrder Γ
      inst✝ : AddCommMonoid R
      α : Type u_7
      β : Type u_8
      s : HahnSeries.SummableFamily Γ R α
      f : Function.Embedding α β
      ⊢ (Set.iUnion fun a => ((fun b => dite (Membership.mem (Set.range ⇑f) b) (fun  …
    -/
    refine s.isPWO_iUnion_support.mono (Set.iUnion_subset fun b g h => ?_)
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_4
      α✝ : Type u_5
      β✝ : Type u_6
      inst✝¹ : PartialOrder Γ
      inst✝ : AddCommMonoid R
      α : Type u_7
      β : Type u_8
      s : HahnSeries.SummableFamily Γ R α
      f : Function.Embedding α β
      b : β
      g : Γ
      h : Membership.mem ((fun b => dite (Membership.mem (Set.range ⇑f) b) (fun h => …
      ⊢ Membership.mem (Set.iUnion fun a => (s a).support) g
    -/
    by_cases hb : b ∈ Set.range f
      /-
        case pos
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        V : Type u_4
        α✝ : Type u_5
        β✝ : Type u_6
        inst✝¹ : PartialOrder Γ
        inst✝ : AddCommMonoid R
        α : Type u_7
        β : Type u_8
        s : HahnSeries.SummableFamily Γ R α
        f : Function.Embedding α β
        b : β
        g : Γ
        h : Membership.mem ((fun b => dite (Membership.mem (Set.range ⇑f) b) (fun h => …
        hb : Membership.mem (Set.range ⇑f) b
        ⊢ Membership.mem (Set.iUnion fun a => (s a).support) g
      -/
    · dsimp only at h
      /-
        case pos
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        V : Type u_4
        α✝ : Type u_5
        β✝ : Type u_6
        inst✝¹ : PartialOrder Γ
        inst✝ : AddCommMonoid R
        α : Type u_7
        β : Type u_8
        s : HahnSeries.SummableFamily Γ R α
        f : Function.Embedding α β
        b : β
        g : Γ
        h : Membership.mem (dite (Membership.mem (Set.range ⇑f) b) (fun h => s (Classi …
        hb : Membership.mem (Set.range ⇑f) b
        ⊢ Membership.mem (Set.iUnion fun a => (s a).support) g
      -/
      rw [dif_pos hb] at h
      /-
        case pos
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        V : Type u_4
        α✝ : Type u_5
        β✝ : Type u_6
        inst✝¹ : PartialOrder Γ
        inst✝ : AddCommMonoid R
        α : Type u_7
        β : Type u_8
        s : HahnSeries.SummableFamily Γ R α
        f : Function.Embedding α β
        b : β
        g : Γ
        hb : Membership.mem (Set.range ⇑f) b
        h : Membership.mem (s (Classical.choose hb)).support g
        ⊢ Membership.mem (Set.iUnion fun a => (s a).support) g
      -/
      exact Set.mem_iUnion.2 ⟨Classical.choose hb, h⟩
      /-
        🎉 no goals
      -/
      /-
        case neg
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        V : Type u_4
        α✝ : Type u_5
        β✝ : Type u_6
        inst✝¹ : PartialOrder Γ
        inst✝ : AddCommMonoid R
        α : Type u_7
        β : Type u_8
        s : HahnSeries.SummableFamily Γ R α
        f : Function.Embedding α β
        b : β
        g : Γ
        h : Membership.mem ((fun b => dite (Membership.mem (Set.range ⇑f) b) (fun h => …
        hb : Not (Membership.mem (Set.range ⇑f) b)
        ⊢ Membership.mem (Set.iUnion fun a => (s a).support) g
      -/
    · simp [-Set.mem_range, dif_neg hb] at h
      /-
        🎉 no goals
      -/
  finite_co_support' g :=
    ((s.finite_co_support g).image f).subset
      (by
        /-
          Γ : Type u_1
          Γ' : Type u_2
          R : Type u_3
          V : Type u_4
          α✝ : Type u_5
          β✝ : Type u_6
          inst✝¹ : PartialOrder Γ
          inst✝ : AddCommMonoid R
          α : Type u_7
          β : Type u_8
          s : HahnSeries.SummableFamily Γ R α
          f : Function.Embedding α β
          g : Γ
          ⊢ HasSubset.Subset (setOf fun a => Ne (((fun b => dite (Membership.mem (Set.ra …
        -/
        intro b h
        /-
          Γ : Type u_1
          Γ' : Type u_2
          R : Type u_3
          V : Type u_4
          α✝ : Type u_5
          β✝ : Type u_6
          inst✝¹ : PartialOrder Γ
          inst✝ : AddCommMonoid R
          α : Type u_7
          β : Type u_8
          s : HahnSeries.SummableFamily Γ R α
          f : Function.Embedding α β
          g : Γ
          b : β
          h : Membership.mem (setOf fun a => Ne (((fun b => dite (Membership.mem (Set.ra …
          ⊢ Membership.mem (Set.image (⇑f) (Function.support fun a => (s a).coeff g)) b
        -/
        by_cases hb : b ∈ Set.range f
          /-
            case pos
            Γ : Type u_1
            Γ' : Type u_2
            R : Type u_3
            V : Type u_4
            α✝ : Type u_5
            β✝ : Type u_6
            inst✝¹ : PartialOrder Γ
            inst✝ : AddCommMonoid R
            α : Type u_7
            β : Type u_8
            s : HahnSeries.SummableFamily Γ R α
            f : Function.Embedding α β
            g : Γ
            b : β
            h : Membership.mem (setOf fun a => Ne (((fun b => dite (Membership.mem (Set.ra …
            hb : Membership.mem (Set.range ⇑f) b
            ⊢ Membership.mem (Set.image (⇑f) (Function.support fun a => (s a).coeff g)) b
          -/
        · simp only [Ne, Set.mem_setOf_eq, dif_pos hb] at h
          /-
            case pos
            Γ : Type u_1
            Γ' : Type u_2
            R : Type u_3
            V : Type u_4
            α✝ : Type u_5
            β✝ : Type u_6
            inst✝¹ : PartialOrder Γ
            inst✝ : AddCommMonoid R
            α : Type u_7
            β : Type u_8
            s : HahnSeries.SummableFamily Γ R α
            f : Function.Embedding α β
            g : Γ
            b : β
            hb : Membership.mem (Set.range ⇑f) b
            h : Not (Eq ((s (Classical.choose hb)).coeff g) 0)
            ⊢ Membership.mem (Set.image (⇑f) (Function.support fun a => (s a).coeff g)) b
          -/
          exact ⟨Classical.choose hb, h, Classical.choose_spec hb⟩
          /-
            🎉 no goals
          -/
          /-
            case neg
            Γ : Type u_1
            Γ' : Type u_2
            R : Type u_3
            V : Type u_4
            α✝ : Type u_5
            β✝ : Type u_6
            inst✝¹ : PartialOrder Γ
            inst✝ : AddCommMonoid R
            α : Type u_7
            β : Type u_8
            s : HahnSeries.SummableFamily Γ R α
            f : Function.Embedding α β
            g : Γ
            b : β
            h : Membership.mem (setOf fun a => Ne (((fun b => dite (Membership.mem (Set.ra …
            hb : Not (Membership.mem (Set.range ⇑f) b)
            ⊢ Membership.mem (Set.image (⇑f) (Function.support fun a => (s a).coeff g)) b
          -/
        · simp only [Ne, Set.mem_setOf_eq, dif_neg hb, zero_coeff, not_true_eq_false] at h)
          /-
            🎉 no goals
          -/


open Classical in
theorem embDomain_apply :
    s.embDomain f b = if h : b ∈ Set.range f then s (Classical.choose h) else 0 :=
  rfl


@[simp]
theorem embDomain_image : s.embDomain f (f a) = s a := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : AddCommMonoid R
    α : Type u_7
    β : Type u_8
    s : HahnSeries.SummableFamily Γ R α
    f : Function.Embedding α β
    a : α
    ⊢ Eq ((s.embDomain f) (f a)) (s a)
  -/
  rw [embDomain_apply, dif_pos (Set.mem_range_self a)]
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : AddCommMonoid R
    α : Type u_7
    β : Type u_8
    s : HahnSeries.SummableFamily Γ R α
    f : Function.Embedding α β
    a : α
    ⊢ Eq (s (Classical.choose ⋯)) (s a)
  -/
  exact congr rfl (f.injective (Classical.choose_spec (Set.mem_range_self a)))
  /-
    🎉 no goals
  -/


@[simp]
theorem embDomain_notin_range (h : b ∉ Set.range f) : s.embDomain f b = 0 := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : AddCommMonoid R
    α : Type u_7
    β : Type u_8
    s : HahnSeries.SummableFamily Γ R α
    f : Function.Embedding α β
    b : β
    h : Not (Membership.mem (Set.range ⇑f) b)
    ⊢ Eq ((s.embDomain f) b) 0
  -/
  rw [embDomain_apply, dif_neg h]
  /-
    🎉 no goals
  -/


@[simp]
theorem hsum_embDomain : (s.embDomain f).hsum = s.hsum := by
  classical
  ext g
  simp only [hsum_coeff, embDomain_apply, apply_dite HahnSeries.coeff, dite_apply, zero_coeff]
  exact finsum_emb_domain f fun a => (s a).coeff g


theorem support_pow_subset_closure [OrderedCancelAddCommMonoid Γ] [Semiring R] (x : HahnSeries Γ R)
    (n : ℕ) : support (x ^ n) ⊆ AddSubmonoid.closure (support x) := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : OrderedCancelAddCommMonoid Γ
    inst✝ : Semiring R
    x : HahnSeries Γ R
    n : Nat
    ⊢ HasSubset.Subset (HPow.hPow x n).support ↑(AddSubmonoid.closure x.support)
  -/
  induction' n with n ih <;> intro g hn
    /-
      case zero
      Γ : Type u_1
      R : Type u_3
      inst✝¹ : OrderedCancelAddCommMonoid Γ
      inst✝ : Semiring R
      x : HahnSeries Γ R
      g : Γ
      hn : Membership.mem (HPow.hPow x 0).support g
      ⊢ Membership.mem (↑(AddSubmonoid.closure x.support)) g
    -/
  · simp only [pow_zero, mem_support, one_coeff, ne_eq, ite_eq_right_iff, Classical.not_imp] at hn
    /-
      case zero
      Γ : Type u_1
      R : Type u_3
      inst✝¹ : OrderedCancelAddCommMonoid Γ
      inst✝ : Semiring R
      x : HahnSeries Γ R
      g : Γ
      hn : And (Eq g 0) (Not (Eq 1 0))
      ⊢ Membership.mem (↑(AddSubmonoid.closure x.support)) g
    -/
    simp only [hn, SetLike.mem_coe]
    /-
      case zero
      Γ : Type u_1
      R : Type u_3
      inst✝¹ : OrderedCancelAddCommMonoid Γ
      inst✝ : Semiring R
      x : HahnSeries Γ R
      g : Γ
      hn : And (Eq g 0) (Not (Eq 1 0))
      ⊢ Membership.mem (AddSubmonoid.closure x.support) 0
    -/
    exact AddSubmonoid.zero_mem _
    /-
      🎉 no goals
    -/
    /-
      case succ
      Γ : Type u_1
      R : Type u_3
      inst✝¹ : OrderedCancelAddCommMonoid Γ
      inst✝ : Semiring R
      x : HahnSeries Γ R
      n : Nat
      ih : HasSubset.Subset (HPow.hPow x n).support ↑(AddSubmonoid.closure x.support)
      g : Γ
      hn : Membership.mem (HPow.hPow x (HAdd.hAdd n 1)).support g
      ⊢ Membership.mem (↑(AddSubmonoid.closure x.support)) g
    -/
  · obtain ⟨i, hi, j, hj, rfl⟩ := support_mul_subset_add_support hn
    /-
      case succ.intro.intro.intro.intro
      Γ : Type u_1
      R : Type u_3
      inst✝¹ : OrderedCancelAddCommMonoid Γ
      inst✝ : Semiring R
      x : HahnSeries Γ R
      n : Nat
      ih : HasSubset.Subset (HPow.hPow x n).support ↑(AddSubmonoid.closure x.support)
      i : Γ
      hi : Membership.mem (npowRec n x).support i
      j : Γ
      hj : Membership.mem x.support j
      hn : Membership.mem (HPow.hPow x (HAdd.hAdd n 1)).support ((fun x1 x2 => HAdd. …
      ⊢ Membership.mem (↑(AddSubmonoid.closure x.support)) ((fun x1 x2 => HAdd.hAdd  …
    -/
    exact SetLike.mem_coe.2 (AddSubmonoid.add_mem _ (ih hi) (AddSubmonoid.subset_closure hj))
    /-
      🎉 no goals
    -/


theorem isPWO_iUnion_support_powers [LinearOrderedCancelAddCommMonoid Γ] [Semiring R]
    {x : HahnSeries Γ R} (hx : 0 ≤ x.order) :
    (⋃ n : ℕ, (x ^ n).support).IsPWO :=
  (x.isPWO_support'.addSubmonoid_closure
    fun _ hg => le_trans hx (order_le_of_coeff_ne_zero (Function.mem_support.mp hg))).mono
    (Set.iUnion_subset fun n => support_pow_subset_closure x n)


theorem co_support_zero [OrderedCancelAddCommMonoid Γ] [Semiring R] (g : Γ) :
    {a | ¬((0 : HahnSeries Γ R) ^ a).coeff g = 0} ⊆ {0} := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : OrderedCancelAddCommMonoid Γ
    inst✝ : Semiring R
    g : Γ
    ⊢ HasSubset.Subset (setOf fun a => Not (Eq ((HPow.hPow 0 a).coeff g) 0)) (Sing …
  -/
  simp only [Set.subset_singleton_iff, Set.mem_setOf_eq]
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : OrderedCancelAddCommMonoid Γ
    inst✝ : Semiring R
    g : Γ
    ⊢ ∀ (y : Nat), Not (Eq ((HPow.hPow 0 y).coeff g) 0) → Eq y 0
  -/
  intro n hn
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : OrderedCancelAddCommMonoid Γ
    inst✝ : Semiring R
    g : Γ
    n : Nat
    hn : Not (Eq ((HPow.hPow 0 n).coeff g) 0)
    ⊢ Eq n 0
  -/
  by_contra h'
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : OrderedCancelAddCommMonoid Γ
    inst✝ : Semiring R
    g : Γ
    n : Nat
    hn : Not (Eq ((HPow.hPow 0 n).coeff g) 0)
    h' : Not (Eq n 0)
    ⊢ False
  -/
  simp_all only [ne_eq, not_false_eq_true, zero_pow, zero_coeff, not_true_eq_false]
  /-
    🎉 no goals
  -/


theorem pow_finite_co_support {x : HahnSeries Γ R} (hx : 0 < x.orderTop) (g : Γ) :
    Set.Finite {a | ((fun n ↦ x ^ n) a).coeff g ≠ 0} := by
  have hpwo : Set.IsPWO (⋃ n, support (x ^ n)) :=
    isPWO_iUnion_support_powers (zero_le_orderTop_iff.mp <| le_of_lt hx)
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : LinearOrderedCancelAddCommMonoid Γ
    inst✝ : CommRing R
    x : HahnSeries Γ R
    hx : LT.lt 0 x.orderTop
    g : Γ
    hpwo : (Set.iUnion fun n => (HPow.hPow x n).support).IsPWO
    ⊢ (setOf fun a => Ne (((fun n => HPow.hPow x n) a).coeff g) 0).Finite
  -/
  by_cases h0 : x = 0; · exact h0 ▸ Set.Finite.subset (Set.finite_singleton 0) (co_support_zero g)
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : LinearOrderedCancelAddCommMonoid Γ
    inst✝ : CommRing R
    x : HahnSeries Γ R
    hx : LT.lt 0 x.orderTop
    g : Γ
    hpwo : (Set.iUnion fun n => (HPow.hPow x n).support).IsPWO
    h0 : Not (Eq x 0)
    ⊢ (setOf fun a => Ne (((fun n => HPow.hPow x n) a).coeff g) 0).Finite
  -/
  by_cases hg : g ∈ ⋃ n : ℕ, { g | (x ^ n).coeff g ≠ 0 }
  /-
    case pos
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : LinearOrderedCancelAddCommMonoid Γ
    inst✝ : CommRing R
    x : HahnSeries Γ R
    hx : LT.lt 0 x.orderTop
    g : Γ
    hpwo : (Set.iUnion fun n => (HPow.hPow x n).support).IsPWO
    h0 : Not (Eq x 0)
    hg : Membership.mem (Set.iUnion fun n => setOf fun g => Ne ((HPow.hPow x n).co …
    ⊢ (setOf fun a => Ne (((fun n => HPow.hPow x n) a).coeff g) 0).Finite
  -/
  swap; · exact Set.finite_empty.subset fun n hn => hg (Set.mem_iUnion.2 ⟨n, hn⟩)
          /-
            🎉 no goals
          -/
  /-
    case pos
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : LinearOrderedCancelAddCommMonoid Γ
    inst✝ : CommRing R
    x : HahnSeries Γ R
    hx : LT.lt 0 x.orderTop
    g : Γ
    hpwo : (Set.iUnion fun n => (HPow.hPow x n).support).IsPWO
    h0 : Not (Eq x 0)
    hg : Membership.mem (Set.iUnion fun n => setOf fun g => Ne ((HPow.hPow x n).co …
    ⊢ (setOf fun a => Ne (((fun n => HPow.hPow x n) a).coeff g) 0).Finite
  -/
  apply hpwo.isWF.induction hg
  /-
    case pos
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : LinearOrderedCancelAddCommMonoid Γ
    inst✝ : CommRing R
    x : HahnSeries Γ R
    hx : LT.lt 0 x.orderTop
    g : Γ
    hpwo : (Set.iUnion fun n => (HPow.hPow x n).support).IsPWO
    h0 : Not (Eq x 0)
    hg : Membership.mem (Set.iUnion fun n => setOf fun g => Ne ((HPow.hPow x n).co …
    ⊢ ∀ (y : Γ), Membership.mem (Set.iUnion fun n => (HPow.hPow x n).support) y →  …
  -/
  intro y ys hy
  refine ((((addAntidiagonal x.isPWO_support hpwo y).finite_toSet.biUnion
    fun ij hij => hy ij.snd (mem_addAntidiagonal.1 (mem_coe.1 hij)).2.1 ?_).image Nat.succ).union
      (Set.finite_singleton 0)).subset ?_
    /-
      case pos.refine_1
      Γ : Type u_1
      R : Type u_3
      inst✝¹ : LinearOrderedCancelAddCommMonoid Γ
      inst✝ : CommRing R
      x : HahnSeries Γ R
      hx : LT.lt 0 x.orderTop
      g : Γ
      hpwo : (Set.iUnion fun n => (HPow.hPow x n).support).IsPWO
      h0 : Not (Eq x 0)
      hg : Membership.mem (Set.iUnion fun n => setOf fun g => Ne ((HPow.hPow x n).co …
      y : Γ
      ys : Membership.mem (Set.iUnion fun n => (HPow.hPow x n).support) y
      hy : ∀ (z : Γ), Membership.mem (Set.iUnion fun n => (HPow.hPow x n).support) z …
      ij : Prod Γ Γ
      hij : Membership.mem (↑(Finset.addAntidiagonal ⋯ hpwo y)) ij
      ⊢ LT.lt ij.2 y
    -/
  · obtain ⟨hi, _, rfl⟩ := mem_addAntidiagonal.1 (mem_coe.1 hij)
    exact lt_add_of_pos_left ij.2 <| lt_of_lt_of_le ((zero_lt_orderTop_iff h0).mp hx) <|
      order_le_of_coeff_ne_zero <| Function.mem_support.mp hi
    /-
      case pos.refine_2
      Γ : Type u_1
      R : Type u_3
      inst✝¹ : LinearOrderedCancelAddCommMonoid Γ
      inst✝ : CommRing R
      x : HahnSeries Γ R
      hx : LT.lt 0 x.orderTop
      g : Γ
      hpwo : (Set.iUnion fun n => (HPow.hPow x n).support).IsPWO
      h0 : Not (Eq x 0)
      hg : Membership.mem (Set.iUnion fun n => setOf fun g => Ne ((HPow.hPow x n).co …
      y : Γ
      ys : Membership.mem (Set.iUnion fun n => (HPow.hPow x n).support) y
      hy : ∀ (z : Γ), Membership.mem (Set.iUnion fun n => (HPow.hPow x n).support) z …
      ⊢ HasSubset.Subset (setOf fun a => Ne (((fun n => HPow.hPow x n) a).coeff y) 0 …
    -/
  · rintro (_ | n) hn
      /-
        case pos.refine_2.zero
        Γ : Type u_1
        R : Type u_3
        inst✝¹ : LinearOrderedCancelAddCommMonoid Γ
        inst✝ : CommRing R
        x : HahnSeries Γ R
        hx : LT.lt 0 x.orderTop
        g : Γ
        hpwo : (Set.iUnion fun n => (HPow.hPow x n).support).IsPWO
        h0 : Not (Eq x 0)
        hg : Membership.mem (Set.iUnion fun n => setOf fun g => Ne ((HPow.hPow x n).co …
        y : Γ
        ys : Membership.mem (Set.iUnion fun n => (HPow.hPow x n).support) y
        hy : ∀ (z : Γ), Membership.mem (Set.iUnion fun n => (HPow.hPow x n).support) z …
        hn : Membership.mem (setOf fun a => Ne (((fun n => HPow.hPow x n) a).coeff y)  …
        ⊢ Membership.mem (Union.union (Set.image Nat.succ (Set.iUnion fun i => Set.iUn …
      -/
    · exact Set.mem_union_right _ (Set.mem_singleton 0)
      /-
        🎉 no goals
      -/
      /-
        case pos.refine_2.succ
        Γ : Type u_1
        R : Type u_3
        inst✝¹ : LinearOrderedCancelAddCommMonoid Γ
        inst✝ : CommRing R
        x : HahnSeries Γ R
        hx : LT.lt 0 x.orderTop
        g : Γ
        hpwo : (Set.iUnion fun n => (HPow.hPow x n).support).IsPWO
        h0 : Not (Eq x 0)
        hg : Membership.mem (Set.iUnion fun n => setOf fun g => Ne ((HPow.hPow x n).co …
        y : Γ
        ys : Membership.mem (Set.iUnion fun n => (HPow.hPow x n).support) y
        hy : ∀ (z : Γ), Membership.mem (Set.iUnion fun n => (HPow.hPow x n).support) z …
        n : Nat
        hn : Membership.mem (setOf fun a => Ne (((fun n => HPow.hPow x n) a).coeff y)  …
        ⊢ Membership.mem (Union.union (Set.image Nat.succ (Set.iUnion fun i => Set.iUn …
      -/
    · obtain ⟨i, hi, j, hj, rfl⟩ := support_mul_subset_add_support hn
      /-
        case pos.refine_2.succ.intro.intro.intro.intro
        Γ : Type u_1
        R : Type u_3
        inst✝¹ : LinearOrderedCancelAddCommMonoid Γ
        inst✝ : CommRing R
        x : HahnSeries Γ R
        hx : LT.lt 0 x.orderTop
        g : Γ
        hpwo : (Set.iUnion fun n => (HPow.hPow x n).support).IsPWO
        h0 : Not (Eq x 0)
        hg : Membership.mem (Set.iUnion fun n => setOf fun g => Ne ((HPow.hPow x n).co …
        n : Nat
        i : Γ
        hi : Membership.mem (npowRec n x).support i
        j : Γ
        hj : Membership.mem x.support j
        ys : Membership.mem (Set.iUnion fun n => (HPow.hPow x n).support) ((fun x1 x2  …
        hy : ∀ (z : Γ), Membership.mem (Set.iUnion fun n => (HPow.hPow x n).support) z …
        hn : Membership.mem (setOf fun a => Ne (((fun n => HPow.hPow x n) a).coeff ((f …
        ⊢ Membership.mem (Union.union (Set.image Nat.succ (Set.iUnion fun i_1 => Set.i …
      -/
      refine Set.mem_union_left _ ⟨n, Set.mem_iUnion.2 ⟨⟨j, i⟩, Set.mem_iUnion.2 ⟨?_, hi⟩⟩, rfl⟩
      /-
        case pos.refine_2.succ.intro.intro.intro.intro
        Γ : Type u_1
        R : Type u_3
        inst✝¹ : LinearOrderedCancelAddCommMonoid Γ
        inst✝ : CommRing R
        x : HahnSeries Γ R
        hx : LT.lt 0 x.orderTop
        g : Γ
        hpwo : (Set.iUnion fun n => (HPow.hPow x n).support).IsPWO
        h0 : Not (Eq x 0)
        hg : Membership.mem (Set.iUnion fun n => setOf fun g => Ne ((HPow.hPow x n).co …
        n : Nat
        i : Γ
        hi : Membership.mem (npowRec n x).support i
        j : Γ
        hj : Membership.mem x.support j
        ys : Membership.mem (Set.iUnion fun n => (HPow.hPow x n).support) ((fun x1 x2  …
        hy : ∀ (z : Γ), Membership.mem (Set.iUnion fun n => (HPow.hPow x n).support) z …
        hn : Membership.mem (setOf fun a => Ne (((fun n => HPow.hPow x n) a).coeff ((f …
        ⊢ Membership.mem ↑(Finset.addAntidiagonal ⋯ hpwo ((fun x1 x2 => HAdd.hAdd x1 x …
      -/
      simp only [mem_coe, mem_addAntidiagonal, mem_support, ne_eq, Set.mem_iUnion]
      /-
        case pos.refine_2.succ.intro.intro.intro.intro
        Γ : Type u_1
        R : Type u_3
        inst✝¹ : LinearOrderedCancelAddCommMonoid Γ
        inst✝ : CommRing R
        x : HahnSeries Γ R
        hx : LT.lt 0 x.orderTop
        g : Γ
        hpwo : (Set.iUnion fun n => (HPow.hPow x n).support).IsPWO
        h0 : Not (Eq x 0)
        hg : Membership.mem (Set.iUnion fun n => setOf fun g => Ne ((HPow.hPow x n).co …
        n : Nat
        i : Γ
        hi : Membership.mem (npowRec n x).support i
        j : Γ
        hj : Membership.mem x.support j
        ys : Membership.mem (Set.iUnion fun n => (HPow.hPow x n).support) ((fun x1 x2  …
        hy : ∀ (z : Γ), Membership.mem (Set.iUnion fun n => (HPow.hPow x n).support) z …
        hn : Membership.mem (setOf fun a => Ne (((fun n => HPow.hPow x n) a).coeff ((f …
        ⊢ And (Not (Eq (x.coeff j) 0)) (And (Exists fun i_1 => Not (Eq ((HPow.hPow x i …
      -/
      exact ⟨hj, ⟨n, hi⟩, add_comm j i⟩
      /-
        🎉 no goals
      -/


/-- The powers of an element of positive valuation form a summable family. -/
@[simps]
def powers (x : HahnSeries Γ R) (hx : 0 < x.orderTop) : SummableFamily Γ R ℕ where
  toFun n := x ^ n
  isPWO_iUnion_support' := isPWO_iUnion_support_powers (zero_le_orderTop_iff.mp <| le_of_lt hx)
  finite_co_support' g := pow_finite_co_support hx g


@[simp]
theorem coe_powers : ⇑(powers x hx) = HPow.hPow x :=
  rfl


theorem embDomain_succ_smul_powers :
    (x • powers x hx).embDomain ⟨Nat.succ, Nat.succ_injective⟩ =
      powers x hx - ofFinsupp (Finsupp.single 0 1) := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : LinearOrderedCancelAddCommMonoid Γ
    inst✝ : CommRing R
    x : HahnSeries Γ R
    hx : LT.lt 0 x.orderTop
    ⊢ Eq ((HSMul.hSMul x (HahnSeries.SummableFamily.powers x hx)).embDomain { toFu …
  -/
  apply SummableFamily.ext
  /-
    case h
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : LinearOrderedCancelAddCommMonoid Γ
    inst✝ : CommRing R
    x : HahnSeries Γ R
    hx : LT.lt 0 x.orderTop
    ⊢ ∀ (a : Nat), Eq (((HSMul.hSMul x (HahnSeries.SummableFamily.powers x hx)).em …
  -/
  rintro (_ | n)
  · rw [embDomain_notin_range, sub_apply, coe_powers, pow_zero, coe_ofFinsupp,
      Finsupp.single_eq_same, sub_self]
    /-
      case h.zero.h
      Γ : Type u_1
      R : Type u_3
      inst✝¹ : LinearOrderedCancelAddCommMonoid Γ
      inst✝ : CommRing R
      x : HahnSeries Γ R
      hx : LT.lt 0 x.orderTop
      ⊢ Not (Membership.mem (Set.range ⇑{ toFun := Nat.succ, inj' := Nat.succ_inject …
    -/
    rw [Set.mem_range, not_exists]
    /-
      case h.zero.h
      Γ : Type u_1
      R : Type u_3
      inst✝¹ : LinearOrderedCancelAddCommMonoid Γ
      inst✝ : CommRing R
      x : HahnSeries Γ R
      hx : LT.lt 0 x.orderTop
      ⊢ ∀ (x : Nat), Not (Eq ({ toFun := Nat.succ, inj' := Nat.succ_injective } x) 0)
    -/
    exact Nat.succ_ne_zero
    /-
      🎉 no goals
    -/
    /-
      case h.succ
      Γ : Type u_1
      R : Type u_3
      inst✝¹ : LinearOrderedCancelAddCommMonoid Γ
      inst✝ : CommRing R
      x : HahnSeries Γ R
      hx : LT.lt 0 x.orderTop
      n : Nat
      ⊢ Eq (((HSMul.hSMul x (HahnSeries.SummableFamily.powers x hx)).embDomain { toF …
    -/
  · refine Eq.trans (embDomain_image _ ⟨Nat.succ, Nat.succ_injective⟩) ?_
    rw [smul_apply, powers_toFun, coe_sub, coe_powers, Pi.sub_apply, coe_ofFinsupp, pow_succ',
      Finsupp.single_eq_of_ne (Nat.zero_ne_add_one n), sub_zero, of_symm_smul_of_eq_mul]


theorem one_sub_self_mul_hsum_powers : (1 - x) * (powers x hx).hsum = 1 := by
  rw [← hsum_smul, sub_smul 1 x (powers x hx), one_smul, hsum_sub, ←
    hsum_embDomain (x • powers x hx) ⟨Nat.succ, Nat.succ_injective⟩, embDomain_succ_smul_powers]
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : LinearOrderedCancelAddCommMonoid Γ
    inst✝ : CommRing R
    x : HahnSeries Γ R
    hx : LT.lt 0 x.orderTop
    ⊢ Eq (HSub.hSub (HahnSeries.SummableFamily.powers x hx).hsum (HSub.hSub (HahnS …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem unit_aux (x : HahnSeries Γ R) {r : R} (hr : r * x.leadingCoeff = 1) :
    0 < (1 - single (-x.order) r * x).orderTop := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝² : LinearOrderedAddCommGroup Γ
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    x : HahnSeries Γ R
    r : R
    hr : Eq (HMul.hMul r x.leadingCoeff) 1
    ⊢ LT.lt 0 (HSub.hSub 1 (HMul.hMul ((HahnSeries.single (Neg.neg x.order)) r) x) …
  -/
  by_cases hx : x = 0; · simp_all [hx]
                         /-
                           🎉 no goals
                         -/
  have hrz : r ≠ 0 := by
    intro h
    rw [h, zero_mul] at hr
    exact (zero_ne_one' R) hr
  /-
    case neg
    Γ : Type u_1
    R : Type u_3
    inst✝² : LinearOrderedAddCommGroup Γ
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    x : HahnSeries Γ R
    r : R
    hr : Eq (HMul.hMul r x.leadingCoeff) 1
    hx : Not (Eq x 0)
    hrz : Ne r 0
    ⊢ LT.lt 0 (HSub.hSub 1 (HMul.hMul ((HahnSeries.single (Neg.neg x.order)) r) x) …
  -/
  refine lt_of_le_of_ne (le_trans ?_ min_orderTop_le_orderTop_sub) fun h => ?_
    /-
      case neg.refine_1
      Γ : Type u_1
      R : Type u_3
      inst✝² : LinearOrderedAddCommGroup Γ
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      x : HahnSeries Γ R
      r : R
      hr : Eq (HMul.hMul r x.leadingCoeff) 1
      hx : Not (Eq x 0)
      hrz : Ne r 0
      ⊢ LE.le 0 (Min.min (HahnSeries.orderTop 1) (HMul.hMul ((HahnSeries.single (Neg …
    -/
  · refine le_min (by rw [orderTop_one]) ?_
    /-
      case neg.refine_1
      Γ : Type u_1
      R : Type u_3
      inst✝² : LinearOrderedAddCommGroup Γ
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      x : HahnSeries Γ R
      r : R
      hr : Eq (HMul.hMul r x.leadingCoeff) 1
      hx : Not (Eq x 0)
      hrz : Ne r 0
      ⊢ LE.le 0 (HMul.hMul ((HahnSeries.single (Neg.neg x.order)) r) x).orderTop
    -/
    refine le_trans ?_ orderTop_add_orderTop_le_orderTop_mul
    /-
      case neg.refine_1
      Γ : Type u_1
      R : Type u_3
      inst✝² : LinearOrderedAddCommGroup Γ
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      x : HahnSeries Γ R
      r : R
      hr : Eq (HMul.hMul r x.leadingCoeff) 1
      hx : Not (Eq x 0)
      hrz : Ne r 0
      ⊢ LE.le 0 (HAdd.hAdd ((HahnSeries.single (Neg.neg x.order)) r).orderTop x.orde …
    -/
    by_cases h : x = 0; · simp [h]
                          /-
                            🎉 no goals
                          -/
    rw [← order_eq_orderTop_of_ne h, orderTop_single
      (fun _ => by simp_all only [zero_mul, zero_ne_one]), ← @WithTop.coe_add,
      WithTop.coe_nonneg, neg_add_cancel]
    /-
      case neg.refine_2
      Γ : Type u_1
      R : Type u_3
      inst✝² : LinearOrderedAddCommGroup Γ
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      x : HahnSeries Γ R
      r : R
      hr : Eq (HMul.hMul r x.leadingCoeff) 1
      hx : Not (Eq x 0)
      hrz : Ne r 0
      h : Eq 0 (HSub.hSub 1 (HMul.hMul ((HahnSeries.single (Neg.neg x.order)) r) x)) …
      ⊢ False
    -/
  · apply coeff_orderTop_ne h.symm
    simp only [C_apply, single_mul_single, zero_add, mul_one, sub_coeff', Pi.sub_apply, one_coeff,
      ↓reduceIte]
    /-
      case neg.refine_2
      Γ : Type u_1
      R : Type u_3
      inst✝² : LinearOrderedAddCommGroup Γ
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      x : HahnSeries Γ R
      r : R
      hr : Eq (HMul.hMul r x.leadingCoeff) 1
      hx : Not (Eq x 0)
      hrz : Ne r 0
      h : Eq 0 (HSub.hSub 1 (HMul.hMul ((HahnSeries.single (Neg.neg x.order)) r) x)) …
      ⊢ Eq (HSub.hSub 1 ((HMul.hMul ((HahnSeries.single (Neg.neg x.order)) r) x).coe …
    -/
    have hrc := mul_coeff_order_add_order ((single (-x.order)) r) x
    /-
      case neg.refine_2
      Γ : Type u_1
      R : Type u_3
      inst✝² : LinearOrderedAddCommGroup Γ
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      x : HahnSeries Γ R
      r : R
      hr : Eq (HMul.hMul r x.leadingCoeff) 1
      hx : Not (Eq x 0)
      hrz : Ne r 0
      h : Eq 0 (HSub.hSub 1 (HMul.hMul ((HahnSeries.single (Neg.neg x.order)) r) x)) …
      hrc : Eq ((HMul.hMul ((HahnSeries.single (Neg.neg x.order)) r) x).coeff (HAdd. …
      ⊢ Eq (HSub.hSub 1 ((HMul.hMul ((HahnSeries.single (Neg.neg x.order)) r) x).coe …
    -/
    rw [order_single hrz, leadingCoeff_of_single, neg_add_cancel, hr] at hrc
    /-
      case neg.refine_2
      Γ : Type u_1
      R : Type u_3
      inst✝² : LinearOrderedAddCommGroup Γ
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      x : HahnSeries Γ R
      r : R
      hr : Eq (HMul.hMul r x.leadingCoeff) 1
      hx : Not (Eq x 0)
      hrz : Ne r 0
      h : Eq 0 (HSub.hSub 1 (HMul.hMul ((HahnSeries.single (Neg.neg x.order)) r) x)) …
      hrc : Eq ((HMul.hMul ((HahnSeries.single (Neg.neg x.order)) r) x).coeff 0) 1
      ⊢ Eq (HSub.hSub 1 ((HMul.hMul ((HahnSeries.single (Neg.neg x.order)) r) x).coe …
    -/
    rw [hrc, sub_self]
    /-
      🎉 no goals
    -/


theorem isUnit_iff {x : HahnSeries Γ R} : IsUnit x ↔ IsUnit (x.leadingCoeff) := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝² : LinearOrderedAddCommGroup Γ
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    x : HahnSeries Γ R
    ⊢ Iff (IsUnit x) (IsUnit x.leadingCoeff)
  -/
  constructor
    /-
      case mp
      Γ : Type u_1
      R : Type u_3
      inst✝² : LinearOrderedAddCommGroup Γ
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      x : HahnSeries Γ R
      ⊢ IsUnit x → IsUnit x.leadingCoeff
    -/
  · rintro ⟨⟨u, i, ui, iu⟩, rfl⟩
    refine
      isUnit_of_mul_eq_one (u.leadingCoeff) (i.leadingCoeff)
        ((mul_coeff_order_add_order u i).symm.trans ?_)
    /-
      case mp.intro.mk
      Γ : Type u_1
      R : Type u_3
      inst✝² : LinearOrderedAddCommGroup Γ
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      u i : HahnSeries Γ R
      ui : Eq (HMul.hMul u i) 1
      iu : Eq (HMul.hMul i u) 1
      ⊢ Eq ((HMul.hMul u i).coeff (HAdd.hAdd u.order i.order)) 1
    -/
    rw [ui, one_coeff, if_pos]
    /-
      case mp.intro.mk.hc
      Γ : Type u_1
      R : Type u_3
      inst✝² : LinearOrderedAddCommGroup Γ
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      u i : HahnSeries Γ R
      ui : Eq (HMul.hMul u i) 1
      iu : Eq (HMul.hMul i u) 1
      ⊢ Eq (HAdd.hAdd u.order i.order) 0
    -/
    rw [← order_mul (left_ne_zero_of_mul_eq_one ui) (right_ne_zero_of_mul_eq_one ui), ui, order_one]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      Γ : Type u_1
      R : Type u_3
      inst✝² : LinearOrderedAddCommGroup Γ
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      x : HahnSeries Γ R
      ⊢ IsUnit x.leadingCoeff → IsUnit x
    -/
  · rintro ⟨⟨u, i, ui, iu⟩, h⟩
    /-
      case mpr.intro.mk
      Γ : Type u_1
      R : Type u_3
      inst✝² : LinearOrderedAddCommGroup Γ
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      x : HahnSeries Γ R
      u i : R
      ui : Eq (HMul.hMul u i) 1
      iu : Eq (HMul.hMul i u) 1
      h : Eq (↑{ val := u, inv := i, val_inv := ui, inv_val := iu }) x.leadingCoeff
      ⊢ IsUnit x
    -/
    rw [Units.val_mk] at h
    /-
      case mpr.intro.mk
      Γ : Type u_1
      R : Type u_3
      inst✝² : LinearOrderedAddCommGroup Γ
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      x : HahnSeries Γ R
      u i : R
      ui : Eq (HMul.hMul u i) 1
      iu : Eq (HMul.hMul i u) 1
      h : Eq u x.leadingCoeff
      ⊢ IsUnit x
    -/
    rw [h] at iu
    /-
      case mpr.intro.mk
      Γ : Type u_1
      R : Type u_3
      inst✝² : LinearOrderedAddCommGroup Γ
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      x : HahnSeries Γ R
      u i : R
      ui : Eq (HMul.hMul u i) 1
      iu : Eq (HMul.hMul i x.leadingCoeff) 1
      h : Eq u x.leadingCoeff
      ⊢ IsUnit x
    -/
    have h := SummableFamily.one_sub_self_mul_hsum_powers (unit_aux x iu)
    /-
      case mpr.intro.mk
      Γ : Type u_1
      R : Type u_3
      inst✝² : LinearOrderedAddCommGroup Γ
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      x : HahnSeries Γ R
      u i : R
      ui : Eq (HMul.hMul u i) 1
      iu : Eq (HMul.hMul i x.leadingCoeff) 1
      h✝ : Eq u x.leadingCoeff
      h : Eq (HMul.hMul (HSub.hSub 1 (HSub.hSub 1 (HMul.hMul ((HahnSeries.single (Ne …
      ⊢ IsUnit x
    -/
    rw [sub_sub_cancel] at h
    /-
      case mpr.intro.mk
      Γ : Type u_1
      R : Type u_3
      inst✝² : LinearOrderedAddCommGroup Γ
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      x : HahnSeries Γ R
      u i : R
      ui : Eq (HMul.hMul u i) 1
      iu : Eq (HMul.hMul i x.leadingCoeff) 1
      h✝ : Eq u x.leadingCoeff
      h : Eq (HMul.hMul (HMul.hMul ((HahnSeries.single (Neg.neg x.order)) i) x) (Hah …
      ⊢ IsUnit x
    -/
    exact isUnit_of_mul_isUnit_right (isUnit_of_mul_eq_one _ _ h)
    /-
      🎉 no goals
    -/


open Classical in
instance instField [Field R] : Field (HahnSeries Γ R) where
  __ : IsDomain (HahnSeries Γ R) := inferInstance
  inv x :=
    if x0 : x = 0 then 0
    else
      (single (-x.order)) (x.leadingCoeff)⁻¹ *
        (SummableFamily.powers _ (unit_aux x (inv_mul_cancel₀ (leadingCoeff_ne_iff.mpr x0)))).hsum
  inv_zero := dif_pos rfl
  mul_inv_cancel x x0 := (congr rfl (dif_neg x0)).trans <| by
    have h :=
      SummableFamily.one_sub_self_mul_hsum_powers
        (unit_aux x (inv_mul_cancel₀ (leadingCoeff_ne_iff.mpr x0)))
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_4
      α : Type u_5
      β : Type u_6
      inst✝¹ : LinearOrderedAddCommGroup Γ
      inst✝ : Field R
      x : HahnSeries Γ R
      x0 : Ne x 0
      h : Eq (HMul.hMul (HSub.hSub 1 (HSub.hSub 1 (HMul.hMul ((HahnSeries.single (Ne …
      ⊢ Eq (HMul.hMul x (HMul.hMul ((HahnSeries.single (Neg.neg x.order)) (Inv.inv x …
    -/
    rw [sub_sub_cancel] at h
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_4
      α : Type u_5
      β : Type u_6
      inst✝¹ : LinearOrderedAddCommGroup Γ
      inst✝ : Field R
      x : HahnSeries Γ R
      x0 : Ne x 0
      h : Eq (HMul.hMul (HMul.hMul ((HahnSeries.single (Neg.neg x.order)) (Inv.inv x …
      ⊢ Eq (HMul.hMul x (HMul.hMul ((HahnSeries.single (Neg.neg x.order)) (Inv.inv x …
    -/
    rw [← mul_assoc, mul_comm x, h]
    /-
      🎉 no goals
    -/
  nnqsmul := _
  nnqsmul_def := fun _ _ => rfl
  qsmul := _
  qsmul_def := fun _ _ => rfl


