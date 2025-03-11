/-- A category has all finite limits if every functor `J ⥤ C` with a `FinCategory J`
instance and `J : Type` has a limit.

This is often called 'finitely complete'.
-/
class HasFiniteLimits : Prop where
  /-- `C` has all limits over any type `J` whose objects and morphisms lie in the same universe
  and which has `FinType` objects and morphisms -/
  out (J : Type) [𝒥 : SmallCategory J] [@FinCategory J 𝒥] : @HasLimitsOfShape J 𝒥 C _


instance (priority := 100) hasLimitsOfShape_of_hasFiniteLimits (J : Type w) [SmallCategory J]
    [FinCategory J] [HasFiniteLimits C] : HasLimitsOfShape J C := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : Type w
    inst✝² : CategoryTheory.SmallCategory J
    inst✝¹ : CategoryTheory.FinCategory J
    inst✝ : CategoryTheory.Limits.HasFiniteLimits C
    ⊢ CategoryTheory.Limits.HasLimitsOfShape J C
  -/
  apply @hasLimitsOfShape_of_equivalence _ _ _ _ _ _ (FinCategory.equivAsType J) ?_
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : Type w
    inst✝² : CategoryTheory.SmallCategory J
    inst✝¹ : CategoryTheory.FinCategory J
    inst✝ : CategoryTheory.Limits.HasFiniteLimits C
    ⊢ CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.FinCategory.AsType J) C
  -/
  apply HasFiniteLimits.out
  /-
    🎉 no goals
  -/


lemma hasFiniteLimits_of_hasLimitsOfSize [HasLimitsOfSize.{v', u'} C] :
    HasFiniteLimits C where
  out := fun J hJ hJ' =>
    haveI := hasLimitsOfSizeShrink.{0, 0} C
    let F := @FinCategory.equivAsType J (@FinCategory.fintypeObj J hJ hJ') hJ hJ'
    @hasLimitsOfShape_of_equivalence (@FinCategory.AsType J (@FinCategory.fintypeObj J hJ hJ'))
    (@FinCategory.categoryAsType J (@FinCategory.fintypeObj J hJ hJ') hJ hJ') _ _ J hJ F _


/-- If `C` has all limits, it has finite limits. -/
instance (priority := 100) hasFiniteLimits_of_hasLimits [HasLimits C] : HasFiniteLimits C :=
  hasFiniteLimits_of_hasLimitsOfSize C


instance (priority := 90) hasFiniteLimits_of_hasLimitsOfSize₀ [HasLimitsOfSize.{0, 0} C] :
    HasFiniteLimits C :=
  hasFiniteLimits_of_hasLimitsOfSize C


/-- We can always derive `HasFiniteLimits C` by providing limits at an
arbitrary universe. -/
theorem hasFiniteLimits_of_hasFiniteLimits_of_size
    (h : ∀ (J : Type w) {𝒥 : SmallCategory J} (_ : @FinCategory J 𝒥), HasLimitsOfShape J C) :
    HasFiniteLimits C where
  out := fun J hJ hhJ => by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      h : ∀ (J : Type w) {𝒥 : CategoryTheory.SmallCategory J}, CategoryTheory.FinCat …
      J : Type
      hJ : CategoryTheory.SmallCategory J
      hhJ : CategoryTheory.FinCategory J
      ⊢ CategoryTheory.Limits.HasLimitsOfShape J C
    -/
    haveI := h (ULiftHom.{w} (ULift.{w} J)) <| @CategoryTheory.finCategoryUlift J hJ hhJ
    have l : @Equivalence J (ULiftHom (ULift J)) hJ
                          (@ULiftHom.category (ULift J) (@uliftCategory J hJ)) :=
      @ULiftHomULiftCategory.equiv J hJ
    apply @hasLimitsOfShape_of_equivalence (ULiftHom (ULift J))
      (@ULiftHom.category (ULift J) (@uliftCategory J hJ)) C _ J hJ
      (@Equivalence.symm J hJ (ULiftHom (ULift J))
      (@ULiftHom.category (ULift J) (@uliftCategory J hJ)) l) _
    /- Porting note: tried to factor out (@instCategoryULiftHom (ULift J) (@uliftCategory J hJ)
    but when doing that would then find the instance and say it was not definitionally equal to
    the provided one (the same thing factored out) -/


/-- A category has all finite colimits if every functor `J ⥤ C` with a `FinCategory J`
instance and `J : Type` has a colimit.

This is often called 'finitely cocomplete'.
-/
class HasFiniteColimits : Prop where
  /-- `C` has all colimits over any type `J` whose objects and morphisms lie in the same universe
  and which has `Fintype` objects and morphisms -/
  out (J : Type) [𝒥 : SmallCategory J] [@FinCategory J 𝒥] : @HasColimitsOfShape J 𝒥 C _


instance (priority := 100) hasColimitsOfShape_of_hasFiniteColimits (J : Type w) [SmallCategory J]
    [FinCategory J] [HasFiniteColimits C] : HasColimitsOfShape J C := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : Type w
    inst✝² : CategoryTheory.SmallCategory J
    inst✝¹ : CategoryTheory.FinCategory J
    inst✝ : CategoryTheory.Limits.HasFiniteColimits C
    ⊢ CategoryTheory.Limits.HasColimitsOfShape J C
  -/
  refine @hasColimitsOfShape_of_equivalence _ _ _ _ _ _ (FinCategory.equivAsType J) ?_
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : Type w
    inst✝² : CategoryTheory.SmallCategory J
    inst✝¹ : CategoryTheory.FinCategory J
    inst✝ : CategoryTheory.Limits.HasFiniteColimits C
    ⊢ CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.FinCategory.AsType  …
  -/
  apply HasFiniteColimits.out
  /-
    🎉 no goals
  -/


lemma hasFiniteColimits_of_hasColimitsOfSize [HasColimitsOfSize.{v', u'} C] :
    HasFiniteColimits C where
  out := fun J hJ hJ' =>
    haveI := hasColimitsOfSizeShrink.{0, 0} C
    let F := @FinCategory.equivAsType J (@FinCategory.fintypeObj J hJ hJ') hJ hJ'
    @hasColimitsOfShape_of_equivalence (@FinCategory.AsType J (@FinCategory.fintypeObj J hJ hJ'))
    (@FinCategory.categoryAsType J (@FinCategory.fintypeObj J hJ hJ') hJ hJ') _ _ J hJ F _


instance (priority := 100) hasFiniteColimits_of_hasColimits [HasColimits C] : HasFiniteColimits C :=
  hasFiniteColimits_of_hasColimitsOfSize C


instance (priority := 90) hasFiniteColimits_of_hasColimitsOfSize₀ [HasColimitsOfSize.{0, 0} C] :
    HasFiniteColimits C :=
  hasFiniteColimits_of_hasColimitsOfSize C


/-- We can always derive `HasFiniteColimits C` by providing colimits at an
arbitrary universe. -/
theorem hasFiniteColimits_of_hasFiniteColimits_of_size
    (h : ∀ (J : Type w) {𝒥 : SmallCategory J} (_ : @FinCategory J 𝒥), HasColimitsOfShape J C) :
    HasFiniteColimits C where
  out := fun J hJ hhJ => by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      h : ∀ (J : Type w) {𝒥 : CategoryTheory.SmallCategory J}, CategoryTheory.FinCat …
      J : Type
      hJ : CategoryTheory.SmallCategory J
      hhJ : CategoryTheory.FinCategory J
      ⊢ CategoryTheory.Limits.HasColimitsOfShape J C
    -/
    haveI := h (ULiftHom.{w} (ULift.{w} J)) <| @CategoryTheory.finCategoryUlift J hJ hhJ
    have l : @Equivalence J (ULiftHom (ULift J)) hJ
                           (@ULiftHom.category (ULift J) (@uliftCategory J hJ)) :=
      @ULiftHomULiftCategory.equiv J hJ
    apply @hasColimitsOfShape_of_equivalence (ULiftHom (ULift J))
      (@ULiftHom.category (ULift J) (@uliftCategory J hJ)) C _ J hJ
      (@Equivalence.symm J hJ (ULiftHom (ULift J))
      (@ULiftHom.category (ULift J) (@uliftCategory J hJ)) l) _


instance fintypeWalkingParallelPair : Fintype WalkingParallelPair where
  elems := [WalkingParallelPair.zero, WalkingParallelPair.one].toFinset
                   /-
                     C : Type u
                     inst✝ : CategoryTheory.Category.{v, u} C
                     x : CategoryTheory.Limits.WalkingParallelPair
                     ⊢ Membership.mem (List.cons CategoryTheory.Limits.WalkingParallelPair.zero (Li …
                   -/
                               /-
                                 🎉 no goals
                               -/
  complete x := by cases x <;> simp
                               /-
                                 🎉 no goals
                               -/

-- attribute [local tidy] tactic.case_bash Porting note: no tidy; no case_bash


instance instFintypeWalkingParallelPairHom (j j' : WalkingParallelPair) :
    Fintype (WalkingParallelPairHom j j') where
  elems :=
    WalkingParallelPair.recOn j
      (WalkingParallelPair.recOn j' [WalkingParallelPairHom.id zero].toFinset
        [left, right].toFinset)
      (WalkingParallelPair.recOn j' ∅ [WalkingParallelPairHom.id one].toFinset)
  complete := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      j j' : CategoryTheory.Limits.WalkingParallelPair
      ⊢ ∀ (x : CategoryTheory.Limits.WalkingParallelPairHom j j'), Membership.mem (C …
    -/
                     /-
                       🎉 no goals
                     -/
                     /-
                       🎉 no goals
                     -/
    rintro (_|_) <;> simp
    /-
      case id
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      j : CategoryTheory.Limits.WalkingParallelPair
      ⊢ Membership.mem (CategoryTheory.Limits.WalkingParallelPair.rec (CategoryTheor …
    -/
                /-
                  🎉 no goals
                -/
    cases j <;> simp
                /-
                  🎉 no goals
                -/

instance : FinCategory WalkingParallelPair where
  fintypeObj := fintypeWalkingParallelPair
  fintypeHom := instFintypeWalkingParallelPairHom -- Porting note: could not be inferred


instance fintypeObj [Fintype J] : Fintype (WidePullbackShape J) :=
  inferInstanceAs <| Fintype (Option _)


instance fintypeHom (j j' : WidePullbackShape J) : Fintype (j ⟶ j') where
  elems := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : Type v
      j j' : CategoryTheory.Limits.WidePullbackShape J
      ⊢ Finset (Quiver.Hom j j')
    -/
    cases' j' with j'
      /-
        case none
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        J : Type v
        j : CategoryTheory.Limits.WidePullbackShape J
        ⊢ Finset (Quiver.Hom j Option.none)
      -/
    · cases' j with j
        /-
          case none.none
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          J : Type v
          ⊢ Finset (Quiver.Hom Option.none Option.none)
        -/
      · exact {Hom.id none}
        /-
          🎉 no goals
        -/
        /-
          case none.some
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          J : Type v
          j : J
          ⊢ Finset (Quiver.Hom (Option.some j) Option.none)
        -/
      · exact {Hom.term j}
        /-
          🎉 no goals
        -/
      /-
        case some
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        J : Type v
        j : CategoryTheory.Limits.WidePullbackShape J
        j' : J
        ⊢ Finset (Quiver.Hom j (Option.some j'))
      -/
    · by_cases h : some j' = j
        /-
          case pos
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          J : Type v
          j : CategoryTheory.Limits.WidePullbackShape J
          j' : J
          h : Eq (Option.some j') j
          ⊢ Finset (Quiver.Hom j (Option.some j'))
        -/
      · rw [h]
        /-
          case pos
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          J : Type v
          j : CategoryTheory.Limits.WidePullbackShape J
          j' : J
          h : Eq (Option.some j') j
          ⊢ Finset (Quiver.Hom j j)
        -/
        exact {Hom.id j}
        /-
          🎉 no goals
        -/
        /-
          case neg
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          J : Type v
          j : CategoryTheory.Limits.WidePullbackShape J
          j' : J
          h : Not (Eq (Option.some j') j)
          ⊢ Finset (Quiver.Hom j (Option.some j'))
        -/
      · exact ∅
        /-
          🎉 no goals
        -/
  complete := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : Type v
      j j' : CategoryTheory.Limits.WidePullbackShape J
      ⊢ ∀ (x : Quiver.Hom j j'), Membership.mem (Option.casesOn (motive := fun t =>  …
    -/
    rintro (_|_)
      /-
        case id
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        J : Type v
        j : CategoryTheory.Limits.WidePullbackShape J
        ⊢ Membership.mem (Option.casesOn (motive := fun t => Eq j t → Finset (Quiver.H …
      -/
                  /-
                    🎉 no goals
                  -/
    · cases j <;> simp
                  /-
                    🎉 no goals
                  -/
      /-
        case term
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        J : Type v
        j✝ : J
        ⊢ Membership.mem (Option.casesOn (motive := fun t => Eq Option.none t → Finset …
      -/
    · simp
      /-
        🎉 no goals
      -/


instance fintypeObj [Fintype J] : Fintype (WidePushoutShape J) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    J : Type v
    inst✝ : Fintype J
    ⊢ Fintype (CategoryTheory.Limits.WidePushoutShape J)
  -/
  rw [WidePushoutShape]; infer_instance
                         /-
                           🎉 no goals
                         -/


instance fintypeHom (j j' : WidePushoutShape J) : Fintype (j ⟶ j') where
  elems := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : Type v
      j j' : CategoryTheory.Limits.WidePushoutShape J
      ⊢ Finset (Quiver.Hom j j')
    -/
    cases' j with j
      /-
        case none
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        J : Type v
        j' : CategoryTheory.Limits.WidePushoutShape J
        ⊢ Finset (Quiver.Hom Option.none j')
      -/
    · cases' j' with j'
        /-
          case none.none
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          J : Type v
          ⊢ Finset (Quiver.Hom Option.none Option.none)
        -/
      · exact {Hom.id none}
        /-
          🎉 no goals
        -/
        /-
          case none.some
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          J : Type v
          j' : J
          ⊢ Finset (Quiver.Hom Option.none (Option.some j'))
        -/
      · exact {Hom.init j'}
        /-
          🎉 no goals
        -/
      /-
        case some
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        J : Type v
        j' : CategoryTheory.Limits.WidePushoutShape J
        j : J
        ⊢ Finset (Quiver.Hom (Option.some j) j')
      -/
    · by_cases h : some j = j'
        /-
          case pos
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          J : Type v
          j' : CategoryTheory.Limits.WidePushoutShape J
          j : J
          h : Eq (Option.some j) j'
          ⊢ Finset (Quiver.Hom (Option.some j) j')
        -/
      · rw [h]
        /-
          case pos
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          J : Type v
          j' : CategoryTheory.Limits.WidePushoutShape J
          j : J
          h : Eq (Option.some j) j'
          ⊢ Finset (Quiver.Hom j' j')
        -/
        exact {Hom.id j'}
        /-
          🎉 no goals
        -/
        /-
          case neg
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          J : Type v
          j' : CategoryTheory.Limits.WidePushoutShape J
          j : J
          h : Not (Eq (Option.some j) j')
          ⊢ Finset (Quiver.Hom (Option.some j) j')
        -/
      · exact ∅
        /-
          🎉 no goals
        -/
  complete := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : Type v
      j j' : CategoryTheory.Limits.WidePushoutShape J
      ⊢ ∀ (x : Quiver.Hom j j'), Membership.mem (Option.casesOn (motive := fun t =>  …
    -/
    rintro (_|_)
      /-
        case id
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        J : Type v
        j : CategoryTheory.Limits.WidePushoutShape J
        ⊢ Membership.mem (Option.casesOn (motive := fun t => Eq j t → Finset (Quiver.H …
      -/
                  /-
                    🎉 no goals
                  -/
    · cases j <;> simp
                  /-
                    🎉 no goals
                  -/
      /-
        case init
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        J : Type v
        j✝ : J
        ⊢ Membership.mem (Option.casesOn (motive := fun t => Eq Option.none t → Finset …
      -/
    · simp
      /-
        🎉 no goals
      -/


instance finCategoryWidePullback [Fintype J] : FinCategory (WidePullbackShape J) where
  fintypeHom := WidePullbackShape.fintypeHom


instance finCategoryWidePushout [Fintype J] : FinCategory (WidePushoutShape J) where
  fintypeHom := WidePushoutShape.fintypeHom

-- We can't just made this an `abbreviation`
-- because of https://github.com/leanprover-community/lean/issues/429

/-- `HasFiniteWidePullbacks` represents a choice of wide pullback
for every finite collection of morphisms
-/
class HasFiniteWidePullbacks : Prop where
  /-- `C` has all wide pullbacks any Fintype `J`-/
  out (J : Type) [Finite J] : HasLimitsOfShape (WidePullbackShape J) C


instance hasLimitsOfShape_widePullbackShape (J : Type) [Finite J] [HasFiniteWidePullbacks C] :
    HasLimitsOfShape (WidePullbackShape J) C := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    J✝ : Type v
    J : Type
    inst✝¹ : Finite J
    inst✝ : CategoryTheory.Limits.HasFiniteWidePullbacks C
    ⊢ CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Limits.WidePullbackSh …
  -/
  haveI := @HasFiniteWidePullbacks.out C _ _ J
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    J✝ : Type v
    J : Type
    inst✝¹ : Finite J
    inst✝ : CategoryTheory.Limits.HasFiniteWidePullbacks C
    this : ∀ [inst : Finite J], CategoryTheory.Limits.HasLimitsOfShape (CategoryTh …
    ⊢ CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Limits.WidePullbackSh …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- `HasFiniteWidePushouts` represents a choice of wide pushout
for every finite collection of morphisms
-/
class HasFiniteWidePushouts : Prop where
  /-- `C` has all wide pushouts any Fintype `J`-/
  out (J : Type) [Finite J] : HasColimitsOfShape (WidePushoutShape J) C


instance hasColimitsOfShape_widePushoutShape (J : Type) [Finite J] [HasFiniteWidePushouts C] :
    HasColimitsOfShape (WidePushoutShape J) C := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    J✝ : Type v
    J : Type
    inst✝¹ : Finite J
    inst✝ : CategoryTheory.Limits.HasFiniteWidePushouts C
    ⊢ CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Limits.WidePushoutS …
  -/
  haveI := @HasFiniteWidePushouts.out C _ _ J
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    J✝ : Type v
    J : Type
    inst✝¹ : Finite J
    inst✝ : CategoryTheory.Limits.HasFiniteWidePushouts C
    this : ∀ [inst : Finite J], CategoryTheory.Limits.HasColimitsOfShape (Category …
    ⊢ CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Limits.WidePushoutS …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- Finite wide pullbacks are finite limits, so if `C` has all finite limits,
it also has finite wide pullbacks
-/
instance (priority := 900) hasFiniteWidePullbacks_of_hasFiniteLimits [HasFiniteLimits C] :
    HasFiniteWidePullbacks C :=
                 /-
                   C : Type u
                   inst✝¹ : CategoryTheory.Category.{v, u} C
                   J✝ : Type v
                   inst✝ : CategoryTheory.Limits.HasFiniteLimits C
                   J : Type
                   x✝ : Finite J
                   ⊢ CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Limits.WidePullbackSh …
                 -/
  ⟨fun J _ => by cases nonempty_fintype J; exact HasFiniteLimits.out _⟩
                                           /-
                                             🎉 no goals
                                           -/


/-- Finite wide pushouts are finite colimits, so if `C` has all finite colimits,
it also has finite wide pushouts
-/
instance (priority := 900) hasFiniteWidePushouts_of_has_finite_limits [HasFiniteColimits C] :
    HasFiniteWidePushouts C :=
                 /-
                   C : Type u
                   inst✝¹ : CategoryTheory.Category.{v, u} C
                   J✝ : Type v
                   inst✝ : CategoryTheory.Limits.HasFiniteColimits C
                   J : Type
                   x✝ : Finite J
                   ⊢ CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Limits.WidePushoutS …
                 -/
  ⟨fun J _ => by cases nonempty_fintype J; exact HasFiniteColimits.out _⟩
                                           /-
                                             🎉 no goals
                                           -/


instance fintypeWalkingPair : Fintype WalkingPair where
  elems := {WalkingPair.left, WalkingPair.right}
                   /-
                     C : Type u
                     inst✝ : CategoryTheory.Category.{v, u} C
                     J : Type v
                     x : CategoryTheory.Limits.WalkingPair
                     ⊢ Membership.mem (Insert.insert CategoryTheory.Limits.WalkingPair.left (Single …
                   -/
                               /-
                                 🎉 no goals
                               -/
  complete x := by cases x <;> simp
                               /-
                                 🎉 no goals
                               -/


