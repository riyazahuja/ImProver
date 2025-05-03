instance widePullbackShape_connected (J : Type v₁) : IsConnected (WidePullbackShape J) := by
  /-
    J : Type v₁
    ⊢ CategoryTheory.IsConnected (CategoryTheory.Limits.WidePullbackShape J)
  -/
  apply IsConnected.of_induct
    /-
      case h
      J : Type v₁
      ⊢ ∀ (p : Set (CategoryTheory.Limits.WidePullbackShape J)), Membership.mem p ?j …
    -/
  · introv hp t
    /-
      case h
      J : Type v₁
      p : Set (CategoryTheory.Limits.WidePullbackShape J)
      hp : Membership.mem p ?j₀
      t : ∀ {j₁ j₂ : CategoryTheory.Limits.WidePullbackShape J}, Quiver.Hom j₁ j₂ →  …
      j : CategoryTheory.Limits.WidePullbackShape J
      ⊢ Membership.mem p j
    -/
    cases j
      /-
        case h.none
        J : Type v₁
        p : Set (CategoryTheory.Limits.WidePullbackShape J)
        hp : Membership.mem p ?j₀
        t : ∀ {j₁ j₂ : CategoryTheory.Limits.WidePullbackShape J}, Quiver.Hom j₁ j₂ →  …
        ⊢ Membership.mem p Option.none
      -/
    · exact hp
      /-
        🎉 no goals
      -/
      /-
        case h.some
        J : Type v₁
        p : Set (CategoryTheory.Limits.WidePullbackShape J)
        hp : Membership.mem p Option.none
        t : ∀ {j₁ j₂ : CategoryTheory.Limits.WidePullbackShape J}, Quiver.Hom j₁ j₂ →  …
        val✝ : J
        ⊢ Membership.mem p (Option.some val✝)
      -/
    · rwa [t (WidePullbackShape.Hom.term _)]
      /-
        🎉 no goals
      -/


instance widePushoutShape_connected (J : Type v₁) : IsConnected (WidePushoutShape J) := by
  /-
    J : Type v₁
    ⊢ CategoryTheory.IsConnected (CategoryTheory.Limits.WidePushoutShape J)
  -/
  apply IsConnected.of_induct
    /-
      case h
      J : Type v₁
      ⊢ ∀ (p : Set (CategoryTheory.Limits.WidePushoutShape J)), Membership.mem p ?j₀ …
    -/
  · introv hp t
    /-
      case h
      J : Type v₁
      p : Set (CategoryTheory.Limits.WidePushoutShape J)
      hp : Membership.mem p ?j₀
      t : ∀ {j₁ j₂ : CategoryTheory.Limits.WidePushoutShape J}, Quiver.Hom j₁ j₂ → I …
      j : CategoryTheory.Limits.WidePushoutShape J
      ⊢ Membership.mem p j
    -/
    cases j
      /-
        case h.none
        J : Type v₁
        p : Set (CategoryTheory.Limits.WidePushoutShape J)
        hp : Membership.mem p ?j₀
        t : ∀ {j₁ j₂ : CategoryTheory.Limits.WidePushoutShape J}, Quiver.Hom j₁ j₂ → I …
        ⊢ Membership.mem p Option.none
      -/
    · exact hp
      /-
        🎉 no goals
      -/
      /-
        case h.some
        J : Type v₁
        p : Set (CategoryTheory.Limits.WidePushoutShape J)
        hp : Membership.mem p Option.none
        t : ∀ {j₁ j₂ : CategoryTheory.Limits.WidePushoutShape J}, Quiver.Hom j₁ j₂ → I …
        val✝ : J
        ⊢ Membership.mem p (Option.some val✝)
      -/
    · rwa [← t (WidePushoutShape.Hom.init _)]
      /-
        🎉 no goals
      -/


instance parallelPairInhabited : Inhabited WalkingParallelPair :=
  ⟨WalkingParallelPair.one⟩


instance parallel_pair_connected : IsConnected WalkingParallelPair := by
  /-
    ⊢ CategoryTheory.IsConnected CategoryTheory.Limits.WalkingParallelPair
  -/
  apply IsConnected.of_induct
    /-
      case h
      ⊢ ∀ (p : Set CategoryTheory.Limits.WalkingParallelPair), Membership.mem p ?j₀  …
    -/
  · introv _ t
    /-
      case h
      p : Set CategoryTheory.Limits.WalkingParallelPair
      a✝ : Membership.mem p ?j₀
      t : ∀ {j₁ j₂ : CategoryTheory.Limits.WalkingParallelPair}, Quiver.Hom j₁ j₂ →  …
      j : CategoryTheory.Limits.WalkingParallelPair
      ⊢ Membership.mem p j
    -/
    cases j
      /-
        case h.zero
        p : Set CategoryTheory.Limits.WalkingParallelPair
        a✝ : Membership.mem p ?j₀
        t : ∀ {j₁ j₂ : CategoryTheory.Limits.WalkingParallelPair}, Quiver.Hom j₁ j₂ →  …
        ⊢ Membership.mem p CategoryTheory.Limits.WalkingParallelPair.zero
      -/
    · rwa [t WalkingParallelPairHom.left]
      /-
        🎉 no goals
      -/
      /-
        case h.one
        p : Set CategoryTheory.Limits.WalkingParallelPair
        a✝ : Membership.mem p CategoryTheory.Limits.WalkingParallelPair.one
        t : ∀ {j₁ j₂ : CategoryTheory.Limits.WalkingParallelPair}, Quiver.Hom j₁ j₂ →  …
        ⊢ Membership.mem p CategoryTheory.Limits.WalkingParallelPair.one
      -/
    · assumption
      /-
        🎉 no goals
      -/


/-- (Impl). The obvious natural transformation from (X × K -) to K. -/
@[simps]
def γ₂ {K : J ⥤ C} (X : C) : K ⋙ prod.functor.obj X ⟶ K where app _ := Limits.prod.snd


/-- (Impl). The obvious natural transformation from (X × K -) to X -/
@[simps]
def γ₁ {K : J ⥤ C} (X : C) : K ⋙ prod.functor.obj X ⟶ (Functor.const J).obj X where
  app _ := Limits.prod.fst


/-- (Impl).
Given a cone for (X × K -), produce a cone for K using the natural transformation `γ₂` -/
@[simps]
def forgetCone {X : C} {K : J ⥤ C} (s : Cone (K ⋙ prod.functor.obj X)) : Cone K where
  pt := s.pt
  π := s.π ≫ γ₂ X


/-- The functor `(X × -)` preserves any connected limit.
Note that this functor does not preserve the two most obvious disconnected limits - that is,
`(X × -)` does not preserve products or terminal object, eg `(X ⨯ A) ⨯ (X ⨯ B)` is not isomorphic to
`X ⨯ (A ⨯ B)` and `X ⨯ 1` is not isomorphic to `1`.
-/
lemma prod_preservesConnectedLimits [IsConnected J] (X : C) :
    PreservesLimitsOfShape J (prod.functor.obj X) where
  preservesLimit {K} :=
    { preserves := fun {c} l => ⟨{
          lift := fun s =>
            prod.lift (s.π.app (Classical.arbitrary _) ≫ Limits.prod.fst) (l.lift (forgetCone s))
          fac := fun s j => by
            /-
              C : Type u₂
              inst✝³ : CategoryTheory.Category.{v₂, u₂} C
              inst✝² : CategoryTheory.Limits.HasBinaryProducts C
              J : Type v₂
              inst✝¹ : CategoryTheory.SmallCategory J
              inst✝ : CategoryTheory.IsConnected J
              X : C
              K : CategoryTheory.Functor J C
              c : CategoryTheory.Limits.Cone K
              l : CategoryTheory.Limits.IsLimit c
              s : CategoryTheory.Limits.Cone (K.comp (CategoryTheory.Limits.prod.functor.obj …
              j : J
              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s => CategoryTheory.Limits.prod …
            -/
            apply Limits.prod.hom_ext
              /-
                case h₁
                C : Type u₂
                inst✝³ : CategoryTheory.Category.{v₂, u₂} C
                inst✝² : CategoryTheory.Limits.HasBinaryProducts C
                J : Type v₂
                inst✝¹ : CategoryTheory.SmallCategory J
                inst✝ : CategoryTheory.IsConnected J
                X : C
                K : CategoryTheory.Functor J C
                c : CategoryTheory.Limits.Cone K
                l : CategoryTheory.Limits.IsLimit c
                s : CategoryTheory.Limits.Cone (K.comp (CategoryTheory.Limits.prod.functor.obj …
                j : J
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
              -/
            · erw [assoc, limMap_π, comp_id, limit.lift_π]
              /-
                case h₁
                C : Type u₂
                inst✝³ : CategoryTheory.Category.{v₂, u₂} C
                inst✝² : CategoryTheory.Limits.HasBinaryProducts C
                J : Type v₂
                inst✝¹ : CategoryTheory.SmallCategory J
                inst✝ : CategoryTheory.IsConnected J
                X : C
                K : CategoryTheory.Functor J C
                c : CategoryTheory.Limits.Cone K
                l : CategoryTheory.Limits.IsLimit c
                s : CategoryTheory.Limits.Cone (K.comp (CategoryTheory.Limits.prod.functor.obj …
                j : J
                ⊢ Eq ((CategoryTheory.Limits.BinaryFan.mk (CategoryTheory.CategoryStruct.comp  …
              -/
              exact (nat_trans_from_is_connected (s.π ≫ γ₁ X) j (Classical.arbitrary _)).symm
              /-
                🎉 no goals
              -/
              /-
                case h₂
                C : Type u₂
                inst✝³ : CategoryTheory.Category.{v₂, u₂} C
                inst✝² : CategoryTheory.Limits.HasBinaryProducts C
                J : Type v₂
                inst✝¹ : CategoryTheory.SmallCategory J
                inst✝ : CategoryTheory.IsConnected J
                X : C
                K : CategoryTheory.Functor J C
                c : CategoryTheory.Limits.Cone K
                l : CategoryTheory.Limits.IsLimit c
                s : CategoryTheory.Limits.Cone (K.comp (CategoryTheory.Limits.prod.functor.obj …
                j : J
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
              -/
            · simp [← l.fac (forgetCone s) j]
              /-
                🎉 no goals
              -/
          uniq := fun s m L => by
            /-
              C : Type u₂
              inst✝³ : CategoryTheory.Category.{v₂, u₂} C
              inst✝² : CategoryTheory.Limits.HasBinaryProducts C
              J : Type v₂
              inst✝¹ : CategoryTheory.SmallCategory J
              inst✝ : CategoryTheory.IsConnected J
              X : C
              K : CategoryTheory.Functor J C
              c : CategoryTheory.Limits.Cone K
              l : CategoryTheory.Limits.IsLimit c
              s : CategoryTheory.Limits.Cone (K.comp (CategoryTheory.Limits.prod.functor.obj …
              m : Quiver.Hom s.pt ((CategoryTheory.Limits.prod.functor.obj X).mapCone c).pt
              L : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (((CategoryTheory.Limi …
              ⊢ Eq m ((fun s => CategoryTheory.Limits.prod.lift (CategoryTheory.CategoryStru …
            -/
            apply Limits.prod.hom_ext
              /-
                case h₁
                C : Type u₂
                inst✝³ : CategoryTheory.Category.{v₂, u₂} C
                inst✝² : CategoryTheory.Limits.HasBinaryProducts C
                J : Type v₂
                inst✝¹ : CategoryTheory.SmallCategory J
                inst✝ : CategoryTheory.IsConnected J
                X : C
                K : CategoryTheory.Functor J C
                c : CategoryTheory.Limits.Cone K
                l : CategoryTheory.Limits.IsLimit c
                s : CategoryTheory.Limits.Cone (K.comp (CategoryTheory.Limits.prod.functor.obj …
                m : Quiver.Hom s.pt ((CategoryTheory.Limits.prod.functor.obj X).mapCone c).pt
                L : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (((CategoryTheory.Limi …
                ⊢ Eq (CategoryTheory.CategoryStruct.comp m CategoryTheory.Limits.prod.fst) (Ca …
              -/
            · erw [limit.lift_π, ← L (Classical.arbitrary J), assoc, limMap_π, comp_id]
              /-
                case h₁
                C : Type u₂
                inst✝³ : CategoryTheory.Category.{v₂, u₂} C
                inst✝² : CategoryTheory.Limits.HasBinaryProducts C
                J : Type v₂
                inst✝¹ : CategoryTheory.SmallCategory J
                inst✝ : CategoryTheory.IsConnected J
                X : C
                K : CategoryTheory.Functor J C
                c : CategoryTheory.Limits.Cone K
                l : CategoryTheory.Limits.IsLimit c
                s : CategoryTheory.Limits.Cone (K.comp (CategoryTheory.Limits.prod.functor.obj …
                m : Quiver.Hom s.pt ((CategoryTheory.Limits.prod.functor.obj X).mapCone c).pt
                L : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (((CategoryTheory.Limi …
                ⊢ Eq (CategoryTheory.CategoryStruct.comp m CategoryTheory.Limits.prod.fst) ((C …
              -/
              rfl
              /-
                🎉 no goals
              -/
              /-
                case h₂
                C : Type u₂
                inst✝³ : CategoryTheory.Category.{v₂, u₂} C
                inst✝² : CategoryTheory.Limits.HasBinaryProducts C
                J : Type v₂
                inst✝¹ : CategoryTheory.SmallCategory J
                inst✝ : CategoryTheory.IsConnected J
                X : C
                K : CategoryTheory.Functor J C
                c : CategoryTheory.Limits.Cone K
                l : CategoryTheory.Limits.IsLimit c
                s : CategoryTheory.Limits.Cone (K.comp (CategoryTheory.Limits.prod.functor.obj …
                m : Quiver.Hom s.pt ((CategoryTheory.Limits.prod.functor.obj X).mapCone c).pt
                L : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (((CategoryTheory.Limi …
                ⊢ Eq (CategoryTheory.CategoryStruct.comp m CategoryTheory.Limits.prod.snd) (Ca …
              -/
            · rw [limit.lift_π]
              /-
                case h₂
                C : Type u₂
                inst✝³ : CategoryTheory.Category.{v₂, u₂} C
                inst✝² : CategoryTheory.Limits.HasBinaryProducts C
                J : Type v₂
                inst✝¹ : CategoryTheory.SmallCategory J
                inst✝ : CategoryTheory.IsConnected J
                X : C
                K : CategoryTheory.Functor J C
                c : CategoryTheory.Limits.Cone K
                l : CategoryTheory.Limits.IsLimit c
                s : CategoryTheory.Limits.Cone (K.comp (CategoryTheory.Limits.prod.functor.obj …
                m : Quiver.Hom s.pt ((CategoryTheory.Limits.prod.functor.obj X).mapCone c).pt
                L : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (((CategoryTheory.Limi …
                ⊢ Eq (CategoryTheory.CategoryStruct.comp m CategoryTheory.Limits.prod.snd) ((C …
              -/
              apply l.uniq (forgetCone s)
              /-
                case h₂.x
                C : Type u₂
                inst✝³ : CategoryTheory.Category.{v₂, u₂} C
                inst✝² : CategoryTheory.Limits.HasBinaryProducts C
                J : Type v₂
                inst✝¹ : CategoryTheory.SmallCategory J
                inst✝ : CategoryTheory.IsConnected J
                X : C
                K : CategoryTheory.Functor J C
                c : CategoryTheory.Limits.Cone K
                l : CategoryTheory.Limits.IsLimit c
                s : CategoryTheory.Limits.Cone (K.comp (CategoryTheory.Limits.prod.functor.obj …
                m : Quiver.Hom s.pt ((CategoryTheory.Limits.prod.functor.obj X).mapCone c).pt
                L : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (((CategoryTheory.Limi …
                ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategorySt …
              -/
              intro j
              /-
                case h₂.x
                C : Type u₂
                inst✝³ : CategoryTheory.Category.{v₂, u₂} C
                inst✝² : CategoryTheory.Limits.HasBinaryProducts C
                J : Type v₂
                inst✝¹ : CategoryTheory.SmallCategory J
                inst✝ : CategoryTheory.IsConnected J
                X : C
                K : CategoryTheory.Functor J C
                c : CategoryTheory.Limits.Cone K
                l : CategoryTheory.Limits.IsLimit c
                s : CategoryTheory.Limits.Cone (K.comp (CategoryTheory.Limits.prod.functor.obj …
                m : Quiver.Hom s.pt ((CategoryTheory.Limits.prod.functor.obj X).mapCone c).pt
                L : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (((CategoryTheory.Limi …
                j : J
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp m …
              -/
              simp [← L j] }⟩ }
              /-
                🎉 no goals
              -/


