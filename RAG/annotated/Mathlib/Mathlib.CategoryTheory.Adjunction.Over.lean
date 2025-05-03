/-- In a category with pullbacks, a morphism `f : X ⟶ Y` induces a functor `Over Y ⥤ Over X`,
by pulling back a morphism along `f`. -/
@[simps! (config := { simpRhs := true}) obj_left obj_hom map_left]
def pullback {X Y : C} (f : X ⟶ Y) : Over Y ⥤ Over X where
  obj g := Over.mk (pullback.snd g.hom f)
  map := fun g {h} {k} =>
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X✝ : C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      X Y : C
      f : Quiver.Hom X Y
      g h : CategoryTheory.Over Y
      k : Quiver.Hom g h
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.lift  …
    -/
          /-
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            X✝ : C
            inst✝ : CategoryTheory.Limits.HasPullbacks C
            X Y : C
            f : Quiver.Hom X Y
            g h : CategoryTheory.Over Y
            k : Quiver.Hom g h
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
          -/
    Over.homMk (pullback.lift (pullback.fst _ _ ≫ k.left) (pullback.snd _ _)
          /-
            🎉 no goals
          -/
    /-
      🎉 no goals
    -/
      (by simp [pullback.condition]))


@[deprecated (since := "2024-05-15")]
noncomputable alias Limits.baseChange := Over.pullback


@[deprecated (since := "2024-07-08")]
noncomputable alias baseChange := pullback


/-- `Over.map f` is left adjoint to `Over.pullback f`. -/
@[simps! unit_app counit_app]
def mapPullbackAdj {X Y : C} (f : X ⟶ Y) : Over.map f ⊣ pullback f :=
  Adjunction.mkOfHomEquiv
    { homEquiv := fun x y =>
        { toFun := fun u =>
                                                         /-
                                                           C : Type u
                                                           inst✝¹ : CategoryTheory.Category.{v, u} C
                                                           X✝ : C
                                                           inst✝ : CategoryTheory.Limits.HasPullbacks C
                                                           X Y : C
                                                           f : Quiver.Hom X Y
                                                           x : CategoryTheory.Over X
                                                           y : CategoryTheory.Over Y
                                                           u : Quiver.Hom ((CategoryTheory.Over.map f).obj x) y
                                                           ⊢ Eq (CategoryTheory.CategoryStruct.comp u.left y.hom) (CategoryTheory.Categor …
                                                         -/
                                                         /-
                                                           🎉 no goals
                                                         -/
            Over.homMk (pullback.lift u.left x.hom <| by simp)
            /-
              🎉 no goals
            -/
          invFun := fun v => Over.homMk (v.left ≫ pullback.fst _ _) <| by
            /-
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              X✝ : C
              inst✝ : CategoryTheory.Limits.HasPullbacks C
              X Y : C
              f : Quiver.Hom X Y
              x : CategoryTheory.Over X
              y : CategoryTheory.Over Y
              v : Quiver.Hom x ((CategoryTheory.Over.pullback f).obj y)
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp v …
            -/
            simp [← Over.w v, pullback.condition]
            /-
              🎉 no goals
            -/
                         /-
                           C : Type u
                           inst✝¹ : CategoryTheory.Category.{v, u} C
                           X✝ : C
                           inst✝ : CategoryTheory.Limits.HasPullbacks C
                           X Y : C
                           f : Quiver.Hom X Y
                           x : CategoryTheory.Over X
                           y : CategoryTheory.Over Y
                           ⊢ Function.LeftInverse (fun v => CategoryTheory.Over.homMk (CategoryTheory.Cat …
                         -/
          left_inv := by aesop_cat
                         /-
                           🎉 no goals
                         -/
          right_inv := fun v => by
            /-
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              X✝ : C
              inst✝ : CategoryTheory.Limits.HasPullbacks C
              X Y : C
              f : Quiver.Hom X Y
              x : CategoryTheory.Over X
              y : CategoryTheory.Over Y
              v : Quiver.Hom x ((CategoryTheory.Over.pullback f).obj y)
              ⊢ Eq ((fun u => CategoryTheory.Over.homMk (CategoryTheory.Limits.pullback.lift …
            -/
            ext
            /-
              case h
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              X✝ : C
              inst✝ : CategoryTheory.Limits.HasPullbacks C
              X Y : C
              f : Quiver.Hom X Y
              x : CategoryTheory.Over X
              y : CategoryTheory.Over Y
              v : Quiver.Hom x ((CategoryTheory.Over.pullback f).obj y)
              ⊢ Eq ((fun u => CategoryTheory.Over.homMk (CategoryTheory.Limits.pullback.lift …
            -/
            dsimp
            /-
              case h
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              X✝ : C
              inst✝ : CategoryTheory.Limits.HasPullbacks C
              X Y : C
              f : Quiver.Hom X Y
              x : CategoryTheory.Over X
              y : CategoryTheory.Over Y
              v : Quiver.Hom x ((CategoryTheory.Over.pullback f).obj y)
              ⊢ Eq (CategoryTheory.Limits.pullback.lift (CategoryTheory.CategoryStruct.comp  …
            -/
            ext
              /-
                case h.h₀
                C : Type u
                inst✝¹ : CategoryTheory.Category.{v, u} C
                X✝ : C
                inst✝ : CategoryTheory.Limits.HasPullbacks C
                X Y : C
                f : Quiver.Hom X Y
                x : CategoryTheory.Over X
                y : CategoryTheory.Over Y
                v : Quiver.Hom x ((CategoryTheory.Over.pullback f).obj y)
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.lift  …
              -/
            · simp
              /-
                🎉 no goals
              -/
              /-
                case h.h₁
                C : Type u
                inst✝¹ : CategoryTheory.Category.{v, u} C
                X✝ : C
                inst✝ : CategoryTheory.Limits.HasPullbacks C
                X Y : C
                f : Quiver.Hom X Y
                x : CategoryTheory.Over X
                y : CategoryTheory.Over Y
                v : Quiver.Hom x ((CategoryTheory.Over.pullback f).obj y)
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.lift  …
              -/
            · simpa using (Over.w v).symm } }
              /-
                🎉 no goals
              -/


@[deprecated (since := "2024-07-08")]
noncomputable alias mapAdjunction := mapPullbackAdj


/-- pullback (𝟙 X) : Over X ⥤ Over X is the identity functor. -/
def pullbackId {X : C} : pullback (𝟙 X) ≅ 𝟭 _ :=
  conjugateIsoEquiv (mapPullbackAdj (𝟙 _)) (Adjunction.id (C := Over _)) (Over.mapId _).symm


/-- pullback commutes with composition (up to natural isomorphism). -/
def pullbackComp {X Y Z : C} (f : X ⟶ Y) (g : Y ⟶ Z) :
    pullback (f ≫ g) ≅ pullback g ⋙ pullback f :=
  conjugateIsoEquiv (mapPullbackAdj _) ((mapPullbackAdj _).comp (mapPullbackAdj _))
    (Over.mapComp _ _).symm


instance pullbackIsRightAdjoint {X Y : C} (f : X ⟶ Y) : (pullback f).IsRightAdjoint  :=
  ⟨_, ⟨mapPullbackAdj f⟩⟩


/--
The functor from `C` to `Over X` which sends `Y : C` to `π₁ : X ⨯ Y ⟶ X`, sometimes denoted `X*`.
-/
@[simps! obj_left obj_hom map_left]
def star [HasBinaryProducts C] : C ⥤ Over X :=
  cofree _ ⋙ coalgebraToOver X


/-- The functor `Over.forget X : Over X ⥤ C` has a right adjoint given by `star X`.

Note that the binary products assumption is necessary: the existence of a right adjoint to
`Over.forget X` is equivalent to the existence of each binary product `X ⨯ -`.
-/
def forgetAdjStar [HasBinaryProducts C] : forget X ⊣ star X :=
  (coalgebraEquivOver X).symm.toAdjunction.comp (adj _)


/-- Note that the binary products assumption is necessary: the existence of a right adjoint to
`Over.forget X` is equivalent to the existence of each binary product `X ⨯ -`.
-/
instance [HasBinaryProducts C] : (forget X).IsLeftAdjoint  :=
  ⟨_, ⟨forgetAdjStar X⟩⟩


@[deprecated (since := "2024-05-18")] noncomputable alias star := Over.star


@[deprecated (since := "2024-05-18")] noncomputable alias forgetAdjStar := Over.forgetAdjStar


/-- When `C` has pushouts, a morphism `f : X ⟶ Y` induces a functor `Under X ⥤ Under Y`,
by pushing a morphism forward along `f`. -/
@[simps]
def pushout {X Y : C} (f : X ⟶ Y) : Under X ⥤ Under Y where
  obj x := Under.mk (pushout.inr x.hom f)
  map := fun x {x'} {u} =>
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X✝ : C
      inst✝ : CategoryTheory.Limits.HasPushouts C
      X Y : C
      f : Quiver.Hom X Y
      x x' : CategoryTheory.Under X
      u : Quiver.Hom x x'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun x => CategoryTheory.Under.mk (C …
    -/
          /-
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            X✝ : C
            inst✝ : CategoryTheory.Limits.HasPushouts C
            X Y : C
            f : Quiver.Hom X Y
            x x' : CategoryTheory.Under X
            u : Quiver.Hom x x'
            ⊢ Eq (CategoryTheory.CategoryStruct.comp x.hom (CategoryTheory.CategoryStruct. …
          -/
    Under.homMk (pushout.desc (u.right ≫ pushout.inl _ _) (pushout.inr _ _)
          /-
            🎉 no goals
          -/
    /-
      🎉 no goals
    -/
      (by simp [← pushout.condition]))


/-- `Under.pushout f` is left adjoint to `Under.map f`. -/
@[simps! unit_app counit_app]
def mapPushoutAdj {X Y : C} (f : X ⟶ Y) : pushout f ⊣ map f :=
  Adjunction.mkOfHomEquiv {
    homEquiv := fun x y => {
      toFun := fun u => Under.homMk (pushout.inl _ _ ≫ u.right) <| by
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          X✝ : C
          inst✝ : CategoryTheory.Limits.HasPushouts C
          X Y : C
          f : Quiver.Hom X Y
          x : CategoryTheory.Under X
          y : CategoryTheory.Under Y
          u : Quiver.Hom ((CategoryTheory.Under.pushout f).obj x) y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp x.hom (CategoryTheory.CategoryStruct. …
        -/
        simp only [map_obj_hom]
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          X✝ : C
          inst✝ : CategoryTheory.Limits.HasPushouts C
          X Y : C
          f : Quiver.Hom X Y
          x : CategoryTheory.Under X
          y : CategoryTheory.Under Y
          u : Quiver.Hom ((CategoryTheory.Under.pushout f).obj x) y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp x.hom (CategoryTheory.CategoryStruct. …
        -/
        rw [← Under.w u]
        simp only [Functor.const_obj_obj, map_obj_right, Functor.id_obj, pushout_obj, mk_right,
          mk_hom]
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          X✝ : C
          inst✝ : CategoryTheory.Limits.HasPushouts C
          X Y : C
          f : Quiver.Hom X Y
          x : CategoryTheory.Under X
          y : CategoryTheory.Under Y
          u : Quiver.Hom ((CategoryTheory.Under.pushout f).obj x) y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp x.hom (CategoryTheory.CategoryStruct. …
        -/
        rw [← assoc, ← assoc, pushout.condition]
        /-
          🎉 no goals
        -/
                                                                       /-
                                                                         C : Type u
                                                                         inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                         X✝ : C
                                                                         inst✝ : CategoryTheory.Limits.HasPushouts C
                                                                         X Y : C
                                                                         f : Quiver.Hom X Y
                                                                         x : CategoryTheory.Under X
                                                                         y : CategoryTheory.Under Y
                                                                         v : Quiver.Hom x ((CategoryTheory.Under.map f).obj y)
                                                                         ⊢ Eq (CategoryTheory.CategoryStruct.comp x.hom v.right) (CategoryTheory.Catego …
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
      invFun := fun v => Under.homMk (pushout.desc v.right y.hom <| by simp)
                         /-
                           🎉 no goals
                         -/
      left_inv := fun u => by
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          X✝ : C
          inst✝ : CategoryTheory.Limits.HasPushouts C
          X Y : C
          f : Quiver.Hom X Y
          x : CategoryTheory.Under X
          y : CategoryTheory.Under Y
          u : Quiver.Hom ((CategoryTheory.Under.pushout f).obj x) y
          ⊢ Eq ((fun v => CategoryTheory.Under.homMk (CategoryTheory.Limits.pushout.desc …
        -/
        ext
        /-
          case h
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          X✝ : C
          inst✝ : CategoryTheory.Limits.HasPushouts C
          X Y : C
          f : Quiver.Hom X Y
          x : CategoryTheory.Under X
          y : CategoryTheory.Under Y
          u : Quiver.Hom ((CategoryTheory.Under.pushout f).obj x) y
          ⊢ Eq ((fun v => CategoryTheory.Under.homMk (CategoryTheory.Limits.pushout.desc …
        -/
        dsimp
        /-
          case h
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          X✝ : C
          inst✝ : CategoryTheory.Limits.HasPushouts C
          X Y : C
          f : Quiver.Hom X Y
          x : CategoryTheory.Under X
          y : CategoryTheory.Under Y
          u : Quiver.Hom ((CategoryTheory.Under.pushout f).obj x) y
          ⊢ Eq (CategoryTheory.Limits.pushout.desc (CategoryTheory.CategoryStruct.comp ( …
        -/
        ext
          /-
            case h.h₀
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            X✝ : C
            inst✝ : CategoryTheory.Limits.HasPushouts C
            X Y : C
            f : Quiver.Hom X Y
            x : CategoryTheory.Under X
            y : CategoryTheory.Under Y
            u : Quiver.Hom ((CategoryTheory.Under.pushout f).obj x) y
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inl x. …
          -/
        · simp
          /-
            🎉 no goals
          -/
          /-
            case h.h₁
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            X✝ : C
            inst✝ : CategoryTheory.Limits.HasPushouts C
            X Y : C
            f : Quiver.Hom X Y
            x : CategoryTheory.Under X
            y : CategoryTheory.Under Y
            u : Quiver.Hom ((CategoryTheory.Under.pushout f).obj x) y
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inr x. …
          -/
        · simpa using (Under.w u).symm
          /-
            🎉 no goals
          -/
                      /-
                        C : Type u
                        inst✝¹ : CategoryTheory.Category.{v, u} C
                        X✝ : C
                        inst✝ : CategoryTheory.Limits.HasPushouts C
                        X Y : C
                        f : Quiver.Hom X Y
                        x : CategoryTheory.Under X
                        y : CategoryTheory.Under Y
                        ⊢ Function.RightInverse (fun v => CategoryTheory.Under.homMk (CategoryTheory.L …
                      -/
      right_inv := by aesop_cat
                      /-
                        🎉 no goals
                      -/
    }
  }


/-- pushout (𝟙 X) : Under X ⥤ Under X is the identity functor. -/
def pushoutId {X : C} : pushout (𝟙 X) ≅ 𝟭 _ :=
  (conjugateIsoEquiv (Adjunction.id (C := Under _)) (mapPushoutAdj (𝟙 _)) ).symm
    (Under.mapId X).symm


/-- pushout commutes with composition (up to natural isomorphism). -/
def pullbackComp {X Y Z : C} (f : X ⟶ Y) (g : Y ⟶ Z) : pushout (f ≫ g) ≅ pushout f ⋙ pushout g :=
  (conjugateIsoEquiv ((mapPushoutAdj _).comp (mapPushoutAdj _)) (mapPushoutAdj _) ).symm
    (mapComp f g).symm


instance pushoutIsLeftAdjoint {X Y : C} (f : X ⟶ Y) : (pushout f).IsLeftAdjoint  :=
  ⟨_, ⟨mapPushoutAdj f⟩⟩


