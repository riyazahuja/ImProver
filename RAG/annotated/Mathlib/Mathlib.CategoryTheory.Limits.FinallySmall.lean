/-- A category is `FinallySmall.{w}` if there is a final functor from a `w`-small category. -/
class FinallySmall : Prop where
  /-- There is a final functor from a small category. -/
  final_smallCategory : ∃ (S : Type w) (_ : SmallCategory S) (F : S ⥤ J), Final F


/-- Constructor for `FinallySmall C` from an explicit small category witness. -/
theorem FinallySmall.mk' {J : Type u} [Category.{v} J] {S : Type w} [SmallCategory S]
    (F : S ⥤ J) [Final F] : FinallySmall.{w} J :=
  ⟨S, _, F, inferInstance⟩


/-- An arbitrarily chosen small model for a finally small category. -/
def FinalModel [FinallySmall.{w} J] : Type w :=
  Classical.choose (@FinallySmall.final_smallCategory J _ _)


noncomputable instance smallCategoryFinalModel [FinallySmall.{w} J] :
    SmallCategory (FinalModel J) :=
  Classical.choose (Classical.choose_spec (@FinallySmall.final_smallCategory J _ _))


/-- An arbitrarily chosen final functor `FinalModel J ⥤ J`. -/
noncomputable def fromFinalModel [FinallySmall.{w} J] : FinalModel J ⥤ J :=
  Classical.choose (Classical.choose_spec (Classical.choose_spec
    (@FinallySmall.final_smallCategory J _ _)))


instance final_fromFinalModel [FinallySmall.{w} J] : Final (fromFinalModel J) :=
  Classical.choose_spec (Classical.choose_spec (Classical.choose_spec
    (@FinallySmall.final_smallCategory J _ _)))


theorem finallySmall_of_essentiallySmall [EssentiallySmall.{w} J] : FinallySmall.{w} J :=
  FinallySmall.mk' (equivSmallModel.{w} J).inverse


theorem finallySmall_of_final_of_finallySmall [FinallySmall.{w} K] (F : K ⥤ J) [Final F] :
    FinallySmall.{w} J :=
  suffices Final ((fromFinalModel K) ⋙ F) from .mk' ((fromFinalModel K) ⋙ F)
  final_comp _ _


theorem finallySmall_of_final_of_essentiallySmall [EssentiallySmall.{w} K] (F : K ⥤ J) [Final F] :
    FinallySmall.{w} J :=
  have := finallySmall_of_essentiallySmall K
  finallySmall_of_final_of_finallySmall F


/-- A category is `InitiallySmall.{w}` if there is an initial functor from a `w`-small category. -/
class InitiallySmall : Prop where
  /-- There is an initial functor from a small category. -/
  initial_smallCategory : ∃ (S : Type w) (_ : SmallCategory S) (F : S ⥤ J), Initial F


/-- Constructor for `InitialSmall C` from an explicit small category witness. -/
theorem InitiallySmall.mk' {J : Type u} [Category.{v} J] {S : Type w} [SmallCategory S]
    (F : S ⥤ J) [Initial F] : InitiallySmall.{w} J :=
  ⟨S, _, F, inferInstance⟩


/-- An arbitrarily chosen small model for an initially small category. -/
def InitialModel [InitiallySmall.{w} J] : Type w :=
  Classical.choose (@InitiallySmall.initial_smallCategory J _ _)


noncomputable instance smallCategoryInitialModel [InitiallySmall.{w} J] :
    SmallCategory (InitialModel J) :=
  Classical.choose (Classical.choose_spec (@InitiallySmall.initial_smallCategory J _ _))


/-- An arbitrarily chosen initial functor `InitialModel J ⥤ J`. -/
noncomputable def fromInitialModel [InitiallySmall.{w} J] : InitialModel J ⥤ J :=
  Classical.choose (Classical.choose_spec (Classical.choose_spec
    (@InitiallySmall.initial_smallCategory J _ _)))


instance initial_fromInitialModel [InitiallySmall.{w} J] : Initial (fromInitialModel J) :=
  Classical.choose_spec (Classical.choose_spec (Classical.choose_spec
    (@InitiallySmall.initial_smallCategory J _ _)))


theorem initiallySmall_of_essentiallySmall [EssentiallySmall.{w} J] : InitiallySmall.{w} J :=
  InitiallySmall.mk' (equivSmallModel.{w} J).inverse


theorem initiallySmall_of_initial_of_initiallySmall [InitiallySmall.{w} K]
    (F : K ⥤ J) [Initial F] : InitiallySmall.{w} J :=
  suffices Initial ((fromInitialModel K) ⋙ F) from .mk' ((fromInitialModel K) ⋙ F)
  initial_comp _ _


theorem initiallySmall_of_initial_of_essentiallySmall [EssentiallySmall.{w} K]
    (F : K ⥤ J) [Initial F] : InitiallySmall.{w} J :=
  have := initiallySmall_of_essentiallySmall K
  initiallySmall_of_initial_of_initiallySmall F


/-- The converse is true if `J` is filtered, see `finallySmall_of_small_weakly_terminal_set`. -/
theorem FinallySmall.exists_small_weakly_terminal_set [FinallySmall.{w} J] :
    ∃ (s : Set J) (_ : Small.{w} s), ∀ i, ∃ j ∈ s, Nonempty (i ⟶ j) := by
  /-
    J : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} J
    inst✝ : CategoryTheory.FinallySmall J
    ⊢ Exists fun s => Exists fun x => ∀ (i : J), Exists fun j => And (Membership.m …
  -/
  refine ⟨Set.range (fromFinalModel J).obj, inferInstance, fun i => ?_⟩
  /-
    J : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} J
    inst✝ : CategoryTheory.FinallySmall J
    i : J
    ⊢ Exists fun j => And (Membership.mem (Set.range (CategoryTheory.fromFinalMode …
  -/
  obtain ⟨f⟩ : Nonempty (StructuredArrow i (fromFinalModel J)) := IsConnected.is_nonempty
  /-
    case intro
    J : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} J
    inst✝ : CategoryTheory.FinallySmall J
    i : J
    f : CategoryTheory.StructuredArrow i (CategoryTheory.fromFinalModel J)
    ⊢ Exists fun j => And (Membership.mem (Set.range (CategoryTheory.fromFinalMode …
  -/
  exact ⟨(fromFinalModel J).obj f.right, Set.mem_range_self _, ⟨f.hom⟩⟩
  /-
    🎉 no goals
  -/


variable {J} in
theorem finallySmall_of_small_weakly_terminal_set [IsFilteredOrEmpty J] (s : Set J) [Small.{v} s]
    (hs : ∀ i, ∃ j ∈ s, Nonempty (i ⟶ j)) : FinallySmall.{v} J := by
  suffices Functor.Final (fullSubcategoryInclusion (· ∈ s)) from
    finallySmall_of_final_of_essentiallySmall (fullSubcategoryInclusion (· ∈ s))
  /-
    J : Type u
    inst✝² : CategoryTheory.Category.{v, u} J
    inst✝¹ : CategoryTheory.IsFilteredOrEmpty J
    s : Set J
    inst✝ : Small.{v, u} ↑s
    hs : ∀ (i : J), Exists fun j => And (Membership.mem s j) (Nonempty (Quiver.Hom …
    ⊢ (CategoryTheory.fullSubcategoryInclusion fun x => Membership.mem s x).Final
  -/
  refine Functor.final_of_exists_of_isFiltered_of_fullyFaithful _ (fun i => ?_)
  /-
    J : Type u
    inst✝² : CategoryTheory.Category.{v, u} J
    inst✝¹ : CategoryTheory.IsFilteredOrEmpty J
    s : Set J
    inst✝ : Small.{v, u} ↑s
    hs : ∀ (i : J), Exists fun j => And (Membership.mem s j) (Nonempty (Quiver.Hom …
    i : J
    ⊢ Exists fun c => Nonempty (Quiver.Hom i ((CategoryTheory.fullSubcategoryInclu …
  -/
  obtain ⟨j, hj₁, hj₂⟩ := hs i
  /-
    case intro.intro
    J : Type u
    inst✝² : CategoryTheory.Category.{v, u} J
    inst✝¹ : CategoryTheory.IsFilteredOrEmpty J
    s : Set J
    inst✝ : Small.{v, u} ↑s
    hs : ∀ (i : J), Exists fun j => And (Membership.mem s j) (Nonempty (Quiver.Hom …
    i j : J
    hj₁ : Membership.mem s j
    hj₂ : Nonempty (Quiver.Hom i j)
    ⊢ Exists fun c => Nonempty (Quiver.Hom i ((CategoryTheory.fullSubcategoryInclu …
  -/
  exact ⟨⟨j, hj₁⟩, hj₂⟩
  /-
    🎉 no goals
  -/


theorem finallySmall_iff_exists_small_weakly_terminal_set [IsFilteredOrEmpty J] :
    FinallySmall.{v} J ↔ ∃ (s : Set J) (_ : Small.{v} s), ∀ i, ∃ j ∈ s, Nonempty (i ⟶ j) := by
  /-
    J : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} J
    inst✝ : CategoryTheory.IsFilteredOrEmpty J
    ⊢ Iff (CategoryTheory.FinallySmall J) (Exists fun s => Exists fun x => ∀ (i :  …
  -/
  refine ⟨fun _ => FinallySmall.exists_small_weakly_terminal_set _, fun h => ?_⟩
  /-
    J : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} J
    inst✝ : CategoryTheory.IsFilteredOrEmpty J
    h : Exists fun s => Exists fun x => ∀ (i : J), Exists fun j => And (Membership …
    ⊢ CategoryTheory.FinallySmall J
  -/
  rcases h with ⟨s, hs, hs'⟩
  /-
    case intro.intro
    J : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} J
    inst✝ : CategoryTheory.IsFilteredOrEmpty J
    s : Set J
    hs : Small.{v, u} ↑s
    hs' : ∀ (i : J), Exists fun j => And (Membership.mem s j) (Nonempty (Quiver.Ho …
    ⊢ CategoryTheory.FinallySmall J
  -/
  exact finallySmall_of_small_weakly_terminal_set s hs'
  /-
    🎉 no goals
  -/


/-- The converse is true if `J` is cofiltered, see `initiallySmall_of_small_weakly_initial_set`. -/
theorem InitiallySmall.exists_small_weakly_initial_set [InitiallySmall.{w} J] :
    ∃ (s : Set J) (_ : Small.{w} s), ∀ i, ∃ j ∈ s, Nonempty (j ⟶ i) := by
  /-
    J : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} J
    inst✝ : CategoryTheory.InitiallySmall J
    ⊢ Exists fun s => Exists fun x => ∀ (i : J), Exists fun j => And (Membership.m …
  -/
  refine ⟨Set.range (fromInitialModel J).obj, inferInstance, fun i => ?_⟩
  /-
    J : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} J
    inst✝ : CategoryTheory.InitiallySmall J
    i : J
    ⊢ Exists fun j => And (Membership.mem (Set.range (CategoryTheory.fromInitialMo …
  -/
  obtain ⟨f⟩ : Nonempty (CostructuredArrow (fromInitialModel J) i) := IsConnected.is_nonempty
  /-
    case intro
    J : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} J
    inst✝ : CategoryTheory.InitiallySmall J
    i : J
    f : CategoryTheory.CostructuredArrow (CategoryTheory.fromInitialModel J) i
    ⊢ Exists fun j => And (Membership.mem (Set.range (CategoryTheory.fromInitialMo …
  -/
  exact ⟨(fromInitialModel J).obj f.left, Set.mem_range_self _, ⟨f.hom⟩⟩
  /-
    🎉 no goals
  -/


variable {J} in
theorem initiallySmall_of_small_weakly_initial_set [IsCofilteredOrEmpty J] (s : Set J) [Small.{v} s]
    (hs : ∀ i, ∃ j ∈ s, Nonempty (j ⟶ i)) : InitiallySmall.{v} J := by
  suffices Functor.Initial (fullSubcategoryInclusion (· ∈ s)) from
    initiallySmall_of_initial_of_essentiallySmall (fullSubcategoryInclusion (· ∈ s))
  /-
    J : Type u
    inst✝² : CategoryTheory.Category.{v, u} J
    inst✝¹ : CategoryTheory.IsCofilteredOrEmpty J
    s : Set J
    inst✝ : Small.{v, u} ↑s
    hs : ∀ (i : J), Exists fun j => And (Membership.mem s j) (Nonempty (Quiver.Hom …
    ⊢ (CategoryTheory.fullSubcategoryInclusion fun x => Membership.mem s x).Initial
  -/
  refine Functor.initial_of_exists_of_isCofiltered_of_fullyFaithful _ (fun i => ?_)
  /-
    J : Type u
    inst✝² : CategoryTheory.Category.{v, u} J
    inst✝¹ : CategoryTheory.IsCofilteredOrEmpty J
    s : Set J
    inst✝ : Small.{v, u} ↑s
    hs : ∀ (i : J), Exists fun j => And (Membership.mem s j) (Nonempty (Quiver.Hom …
    i : J
    ⊢ Exists fun c => Nonempty (Quiver.Hom ((CategoryTheory.fullSubcategoryInclusi …
  -/
  obtain ⟨j, hj₁, hj₂⟩ := hs i
  /-
    case intro.intro
    J : Type u
    inst✝² : CategoryTheory.Category.{v, u} J
    inst✝¹ : CategoryTheory.IsCofilteredOrEmpty J
    s : Set J
    inst✝ : Small.{v, u} ↑s
    hs : ∀ (i : J), Exists fun j => And (Membership.mem s j) (Nonempty (Quiver.Hom …
    i j : J
    hj₁ : Membership.mem s j
    hj₂ : Nonempty (Quiver.Hom j i)
    ⊢ Exists fun c => Nonempty (Quiver.Hom ((CategoryTheory.fullSubcategoryInclusi …
  -/
  exact ⟨⟨j, hj₁⟩, hj₂⟩
  /-
    🎉 no goals
  -/


theorem initiallySmall_iff_exists_small_weakly_initial_set [IsCofilteredOrEmpty J] :
    InitiallySmall.{v} J ↔ ∃ (s : Set J) (_ : Small.{v} s), ∀ i, ∃ j ∈ s, Nonempty (j ⟶ i) := by
  /-
    J : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} J
    inst✝ : CategoryTheory.IsCofilteredOrEmpty J
    ⊢ Iff (CategoryTheory.InitiallySmall J) (Exists fun s => Exists fun x => ∀ (i  …
  -/
  refine ⟨fun _ => InitiallySmall.exists_small_weakly_initial_set _, fun h => ?_⟩
  /-
    J : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} J
    inst✝ : CategoryTheory.IsCofilteredOrEmpty J
    h : Exists fun s => Exists fun x => ∀ (i : J), Exists fun j => And (Membership …
    ⊢ CategoryTheory.InitiallySmall J
  -/
  rcases h with ⟨s, hs, hs'⟩
  /-
    case intro.intro
    J : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} J
    inst✝ : CategoryTheory.IsCofilteredOrEmpty J
    s : Set J
    hs : Small.{v, u} ↑s
    hs' : ∀ (i : J), Exists fun j => And (Membership.mem s j) (Nonempty (Quiver.Ho …
    ⊢ CategoryTheory.InitiallySmall J
  -/
  exact initiallySmall_of_small_weakly_initial_set s hs'
  /-
    🎉 no goals
  -/


theorem hasColimitsOfShape_of_finallySmall (J : Type u) [Category.{v} J] [FinallySmall.{w} J]
    (C : Type u₁) [Category.{v₁} C] [HasColimitsOfSize.{w, w} C] : HasColimitsOfShape J C :=
  Final.hasColimitsOfShape_of_final (fromFinalModel J)


theorem hasLimitsOfShape_of_initiallySmall (J : Type u) [Category.{v} J] [InitiallySmall.{w} J]
    (C : Type u₁) [Category.{v₁} C] [HasLimitsOfSize.{w, w} C] : HasLimitsOfShape J C :=
  Initial.hasLimitsOfShape_of_initial (fromInitialModel J)


