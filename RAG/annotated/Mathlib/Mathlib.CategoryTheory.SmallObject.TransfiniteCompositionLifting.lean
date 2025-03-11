/-- Given a cocone `c` for a functor `F : J ⥤ C` from a well-ordered type,
and maps `p : X ⟶ Y`, `f : F.obj ⊥ ⟶ X`, `g : c.pt ⟶ Y`, this structure
contains the data of a map `F.obj j ⟶ X` such that `F.map (homOfLE bot_le) ≫ f' = f`
and `f' ≫ p = c.ι.app j ≫ g`. (This implies that the outer square below
commutes, see `SqStruct.w`.)

```
         f
F.obj ⊥ --> X
   |      Λ |
   |   f'╱  |
   v    ╱   |
F.obj j     | p
   |        |
   |        |
   v    g   v
  c.pt ---> Y
```
-/
@[ext]
structure SqStruct (j : J) where
  /-- a morphism `F.obj j ⟶ X` -/
  f' : F.obj j ⟶ X
  w₁ : F.map (homOfLE bot_le) ≫ f' = f := by aesop_cat
  w₂ : f' ≫ p = c.ι.app j ≫ g := by aesop_cat


attribute [reassoc (attr := simp)] w₁ w₂


include sq' in
@[reassoc]
lemma w : f ≫ p = c.ι.app ⊥ ≫ g := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    J : Type w
    inst✝¹ : LinearOrder J
    inst✝ : OrderBot J
    F : CategoryTheory.Functor J C
    c : CategoryTheory.Limits.Cocone F
    X Y : C
    p : Quiver.Hom X Y
    f : Quiver.Hom (F.obj Bot.bot) X
    g : Quiver.Hom c.pt Y
    j : J
    sq' : CategoryTheory.HasLiftingProperty.transfiniteComposition.SqStruct c p f  …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f p) (CategoryTheory.CategoryStruct.c …
  -/
  rw [← sq'.w₁, assoc, sq'.w₂, Cocone.w_assoc]
  /-
    🎉 no goals
  -/


/--
Given `sq' : SqStruct c p f g j`, this is the commutative square
```
               sq'.f'
F.obj j --------------------> X
   |                          |
   |                          |p
   v                      g   v
F.obj (succ j) ---> c.pt ---> Y
```

(Using the lifting property for this square is the key ingredient
in the proof that the left lifting property with respect to `p`
is stable under transfinite composition.) -/
lemma sq [SuccOrder J] :
    CommSq sq'.f' (F.map (homOfLE (Order.le_succ j))) p (c.ι.app _ ≫ g) where
          /-
            C : Type u
            inst✝³ : CategoryTheory.Category.{v, u} C
            J : Type w
            inst✝² : LinearOrder J
            inst✝¹ : OrderBot J
            F : CategoryTheory.Functor J C
            c : CategoryTheory.Limits.Cocone F
            X Y : C
            p : Quiver.Hom X Y
            f : Quiver.Hom (F.obj Bot.bot) X
            g : Quiver.Hom c.pt Y
            j : J
            sq' : CategoryTheory.HasLiftingProperty.transfiniteComposition.SqStruct c p f  …
            inst✝ : SuccOrder J
            ⊢ Eq (CategoryTheory.CategoryStruct.comp sq'.f' p) (CategoryTheory.CategoryStr …
          -/
  w := by simp
          /-
            🎉 no goals
          -/


/-- Auxiliary definition for `sqFunctor`. -/
@[simps]
def map {j' : J} (α : j' ⟶ j) : SqStruct c p f g j' where
  f' := F.map α ≫ sq'.f'
  w₁ := by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      W : CategoryTheory.MorphismProperty C
      J : Type w
      inst✝¹ : LinearOrder J
      inst✝ : OrderBot J
      F : CategoryTheory.Functor J C
      c : CategoryTheory.Limits.Cocone F
      hc : CategoryTheory.Limits.IsColimit c
      X Y : C
      p : Quiver.Hom X Y
      f : Quiver.Hom (F.obj Bot.bot) X
      g : Quiver.Hom c.pt Y
      j : J
      sq' : CategoryTheory.HasLiftingProperty.transfiniteComposition.SqStruct c p f  …
      j' : J
      α : Quiver.Hom j' j
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.homOfLE ⋯)) (C …
    -/
    rw [← F.map_comp_assoc]
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      W : CategoryTheory.MorphismProperty C
      J : Type w
      inst✝¹ : LinearOrder J
      inst✝ : OrderBot J
      F : CategoryTheory.Functor J C
      c : CategoryTheory.Limits.Cocone F
      hc : CategoryTheory.Limits.IsColimit c
      X Y : C
      p : Quiver.Hom X Y
      f : Quiver.Hom (F.obj Bot.bot) X
      g : Quiver.Hom c.pt Y
      j : J
      sq' : CategoryTheory.HasLiftingProperty.transfiniteComposition.SqStruct c p f  …
      j' : J
      α : Quiver.Hom j' j
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.CategoryStruct …
    -/
    exact sq'.w₁
    /-
      🎉 no goals
    -/


/-- The projective system `j ↦ SqStruct c p f g j.unop`. -/
@[simps]
def sqFunctor : Jᵒᵖ ⥤ Type _ where
  obj j := SqStruct c p f g j.unop
  map α sq' := sq'.map α.unop


/-- Auxiliary definition for `transfiniteComposition.wellOrderInductionData`. -/
noncomputable def liftHom : F.obj j ⟶ X :=
  (F.isColimitOfIsWellOrderContinuous j hj).desc
    (Cocone.mk _
      { app := fun i ↦ (s.1 ⟨i⟩).f'
        naturality i i' g := by
          /-
            C : Type u
            inst✝³ : CategoryTheory.Category.{v, u} C
            W : CategoryTheory.MorphismProperty C
            J : Type w
            inst✝² : LinearOrder J
            inst✝¹ : OrderBot J
            F : CategoryTheory.Functor J C
            c : CategoryTheory.Limits.Cocone F
            hc : CategoryTheory.Limits.IsColimit c
            X Y : C
            p : Quiver.Hom X Y
            f : Quiver.Hom (F.obj Bot.bot) X
            g✝ : Quiver.Hom c.pt Y
            inst✝ : F.IsWellOrderContinuous
            j : J
            hj : Order.IsSuccLimit j
            s : ↑(⋯.functor.op.comp (CategoryTheory.HasLiftingProperty.transfiniteComposit …
            i i' : ↑(Set.Iio j)
            g : Quiver.Hom i i'
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.restrictionLT j).map g) ((fun i = …
          -/
          have := congr_arg SqStruct.f' (s.2 g.op)
          /-
            C : Type u
            inst✝³ : CategoryTheory.Category.{v, u} C
            W : CategoryTheory.MorphismProperty C
            J : Type w
            inst✝² : LinearOrder J
            inst✝¹ : OrderBot J
            F : CategoryTheory.Functor J C
            c : CategoryTheory.Limits.Cocone F
            hc : CategoryTheory.Limits.IsColimit c
            X Y : C
            p : Quiver.Hom X Y
            f : Quiver.Hom (F.obj Bot.bot) X
            g✝ : Quiver.Hom c.pt Y
            inst✝ : F.IsWellOrderContinuous
            j : J
            hj : Order.IsSuccLimit j
            s : ↑(⋯.functor.op.comp (CategoryTheory.HasLiftingProperty.transfiniteComposit …
            i i' : ↑(Set.Iio j)
            g : Quiver.Hom i i'
            this : Eq ((⋯.functor.op.comp (CategoryTheory.HasLiftingProperty.transfiniteCo …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.restrictionLT j).map g) ((fun i = …
          -/
          dsimp at this ⊢
          /-
            C : Type u
            inst✝³ : CategoryTheory.Category.{v, u} C
            W : CategoryTheory.MorphismProperty C
            J : Type w
            inst✝² : LinearOrder J
            inst✝¹ : OrderBot J
            F : CategoryTheory.Functor J C
            c : CategoryTheory.Limits.Cocone F
            hc : CategoryTheory.Limits.IsColimit c
            X Y : C
            p : Quiver.Hom X Y
            f : Quiver.Hom (F.obj Bot.bot) X
            g✝ : Quiver.Hom c.pt Y
            inst✝ : F.IsWellOrderContinuous
            j : J
            hj : Order.IsSuccLimit j
            s : ↑(⋯.functor.op.comp (CategoryTheory.HasLiftingProperty.transfiniteComposit …
            i i' : ↑(Set.Iio j)
            g : Quiver.Hom i i'
            this : Eq (CategoryTheory.CategoryStruct.comp (F.map (⋯.functor.map g)) (↑s {  …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map ((Monotone.functor ⋯).map g))  …
          -/
          rw [this, comp_id] })
          /-
            🎉 no goals
          -/


@[reassoc]
lemma liftHom_fac (i : J) (hi : i < j) :
    F.map (homOfLE hi.le) ≫ liftHom hj s = (s.1 ⟨⟨i, hi⟩⟩).f' :=
  (F.isColimitOfIsWellOrderContinuous j hj).fac _ ⟨i, hi⟩


/-- Auxiliary definition for `transfiniteComposition.wellOrderInductionData`. -/
@[simps]
noncomputable def lift : (sqFunctor c p f g).obj (Opposite.op j) where
  f' := liftHom hj s
  w₁ := by
    have h : ⊥ < j := Ne.bot_lt' (by
      rintro rfl
      exact Order.not_isSuccLimit_bot hj)
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      W : CategoryTheory.MorphismProperty C
      J : Type w
      inst✝² : LinearOrder J
      inst✝¹ : OrderBot J
      F : CategoryTheory.Functor J C
      c : CategoryTheory.Limits.Cocone F
      hc : CategoryTheory.Limits.IsColimit c
      X Y : C
      p : Quiver.Hom X Y
      f : Quiver.Hom (F.obj Bot.bot) X
      g : Quiver.Hom c.pt Y
      inst✝ : F.IsWellOrderContinuous
      j : J
      hj : Order.IsSuccLimit j
      s : ↑(⋯.functor.op.comp (CategoryTheory.HasLiftingProperty.transfiniteComposit …
      h : LT.lt Bot.bot j
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.homOfLE ⋯)) (C …
    -/
    rw [liftHom_fac hj s ⊥ h]
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      W : CategoryTheory.MorphismProperty C
      J : Type w
      inst✝² : LinearOrder J
      inst✝¹ : OrderBot J
      F : CategoryTheory.Functor J C
      c : CategoryTheory.Limits.Cocone F
      hc : CategoryTheory.Limits.IsColimit c
      X Y : C
      p : Quiver.Hom X Y
      f : Quiver.Hom (F.obj Bot.bot) X
      g : Quiver.Hom c.pt Y
      inst✝ : F.IsWellOrderContinuous
      j : J
      hj : Order.IsSuccLimit j
      s : ↑(⋯.functor.op.comp (CategoryTheory.HasLiftingProperty.transfiniteComposit …
      h : LT.lt Bot.bot j
      ⊢ Eq (↑s { unop := ⟨Bot.bot, h⟩ }).f' f
    -/
    simpa using (s.1 ⟨⊥, h⟩).w₁
    /-
      🎉 no goals
    -/
  w₂ := (F.isColimitOfIsWellOrderContinuous j hj).hom_ext (fun ⟨i, hij⟩ ↦ by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      W : CategoryTheory.MorphismProperty C
      J : Type w
      inst✝² : LinearOrder J
      inst✝¹ : OrderBot J
      F : CategoryTheory.Functor J C
      c : CategoryTheory.Limits.Cocone F
      hc : CategoryTheory.Limits.IsColimit c
      X Y : C
      p : Quiver.Hom X Y
      f : Quiver.Hom (F.obj Bot.bot) X
      g : Quiver.Hom c.pt Y
      inst✝ : F.IsWellOrderContinuous
      j : J
      hj : Order.IsSuccLimit j
      s : ↑(⋯.functor.op.comp (CategoryTheory.HasLiftingProperty.transfiniteComposit …
      x✝ : ↑(Set.Iio j)
      i : J
      hij : Membership.mem (Set.Iio j) i
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.coconeLT j).ι.app ⟨i, hij⟩) (Cate …
    -/
    have := (s.1 ⟨i, hij⟩).w₂
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      W : CategoryTheory.MorphismProperty C
      J : Type w
      inst✝² : LinearOrder J
      inst✝¹ : OrderBot J
      F : CategoryTheory.Functor J C
      c : CategoryTheory.Limits.Cocone F
      hc : CategoryTheory.Limits.IsColimit c
      X Y : C
      p : Quiver.Hom X Y
      f : Quiver.Hom (F.obj Bot.bot) X
      g : Quiver.Hom c.pt Y
      inst✝ : F.IsWellOrderContinuous
      j : J
      hj : Order.IsSuccLimit j
      s : ↑(⋯.functor.op.comp (CategoryTheory.HasLiftingProperty.transfiniteComposit …
      x✝ : ↑(Set.Iio j)
      i : J
      hij : Membership.mem (Set.Iio j) i
      this : Eq (CategoryTheory.CategoryStruct.comp (↑s { unop := ⟨i, hij⟩ }).f' p)  …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.coconeLT j).ι.app ⟨i, hij⟩) (Cate …
    -/
    dsimp at this ⊢
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      W : CategoryTheory.MorphismProperty C
      J : Type w
      inst✝² : LinearOrder J
      inst✝¹ : OrderBot J
      F : CategoryTheory.Functor J C
      c : CategoryTheory.Limits.Cocone F
      hc : CategoryTheory.Limits.IsColimit c
      X Y : C
      p : Quiver.Hom X Y
      f : Quiver.Hom (F.obj Bot.bot) X
      g : Quiver.Hom c.pt Y
      inst✝ : F.IsWellOrderContinuous
      j : J
      hj : Order.IsSuccLimit j
      s : ↑(⋯.functor.op.comp (CategoryTheory.HasLiftingProperty.transfiniteComposit …
      x✝ : ↑(Set.Iio j)
      i : J
      hij : Membership.mem (Set.Iio j) i
      this : Eq (CategoryTheory.CategoryStruct.comp (↑s { unop := ⟨i, hij⟩ }).f' p)  …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.homOfLE ⋯)) (C …
    -/
    rw [liftHom_fac_assoc _ _ _ hij, this, Cocone.w_assoc])
    /-
      🎉 no goals
    -/


lemma map_lift {i : J} (hij : i < j) :
    (lift hj s).map (homOfLE hij.le) = s.1 ⟨⟨i, hij⟩⟩ := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : Type w
    inst✝² : LinearOrder J
    inst✝¹ : OrderBot J
    F : CategoryTheory.Functor J C
    c : CategoryTheory.Limits.Cocone F
    X Y : C
    p : Quiver.Hom X Y
    f : Quiver.Hom (F.obj Bot.bot) X
    g : Quiver.Hom c.pt Y
    inst✝ : F.IsWellOrderContinuous
    j : J
    hj : Order.IsSuccLimit j
    s : ↑(⋯.functor.op.comp (CategoryTheory.HasLiftingProperty.transfiniteComposit …
    i : J
    hij : LT.lt i j
    ⊢ Eq (CategoryTheory.HasLiftingProperty.transfiniteComposition.SqStruct.map (C …
  -/
  ext
  /-
    case f'
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : Type w
    inst✝² : LinearOrder J
    inst✝¹ : OrderBot J
    F : CategoryTheory.Functor J C
    c : CategoryTheory.Limits.Cocone F
    X Y : C
    p : Quiver.Hom X Y
    f : Quiver.Hom (F.obj Bot.bot) X
    g : Quiver.Hom c.pt Y
    inst✝ : F.IsWellOrderContinuous
    j : J
    hj : Order.IsSuccLimit j
    s : ↑(⋯.functor.op.comp (CategoryTheory.HasLiftingProperty.transfiniteComposit …
    i : J
    hij : LT.lt i j
    ⊢ Eq (CategoryTheory.HasLiftingProperty.transfiniteComposition.SqStruct.map (C …
  -/
  apply liftHom_fac
  /-
    🎉 no goals
  -/


open wellOrderInductionData in
/-- The projective system `sqFunctor c p f g` has a `WellOrderInductionData` structure. -/
noncomputable def wellOrderInductionData :
    (sqFunctor c p f g).WellOrderInductionData where
  succ j hj sq' :=
    have := hF j hj
    { f' := sq'.sq.lift
      w₁ := by
        /-
          C : Type u
          inst✝⁵ : CategoryTheory.Category.{v, u} C
          W : CategoryTheory.MorphismProperty C
          J : Type w
          inst✝⁴ : LinearOrder J
          inst✝³ : OrderBot J
          F : CategoryTheory.Functor J C
          c : CategoryTheory.Limits.Cocone F
          hc : CategoryTheory.Limits.IsColimit c
          X Y : C
          p : Quiver.Hom X Y
          f : Quiver.Hom (F.obj Bot.bot) X
          g : Quiver.Hom c.pt Y
          inst✝² : F.IsWellOrderContinuous
          inst✝¹ : SuccOrder J
          inst✝ : WellFoundedLT J
          hF : ∀ (j : J), Not (IsMax j) → CategoryTheory.HasLiftingProperty (F.map (Cate …
          j : J
          hj : Not (IsMax j)
          sq' : (CategoryTheory.HasLiftingProperty.transfiniteComposition.sqFunctor c p  …
          this : CategoryTheory.HasLiftingProperty (F.map (CategoryTheory.homOfLE ⋯)) p
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.homOfLE ⋯)) ⋯. …
        -/
        dsimp
        /-
          C : Type u
          inst✝⁵ : CategoryTheory.Category.{v, u} C
          W : CategoryTheory.MorphismProperty C
          J : Type w
          inst✝⁴ : LinearOrder J
          inst✝³ : OrderBot J
          F : CategoryTheory.Functor J C
          c : CategoryTheory.Limits.Cocone F
          hc : CategoryTheory.Limits.IsColimit c
          X Y : C
          p : Quiver.Hom X Y
          f : Quiver.Hom (F.obj Bot.bot) X
          g : Quiver.Hom c.pt Y
          inst✝² : F.IsWellOrderContinuous
          inst✝¹ : SuccOrder J
          inst✝ : WellFoundedLT J
          hF : ∀ (j : J), Not (IsMax j) → CategoryTheory.HasLiftingProperty (F.map (Cate …
          j : J
          hj : Not (IsMax j)
          sq' : (CategoryTheory.HasLiftingProperty.transfiniteComposition.sqFunctor c p  …
          this : CategoryTheory.HasLiftingProperty (F.map (CategoryTheory.homOfLE ⋯)) p
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.homOfLE ⋯)) ⋯. …
        -/
        simp only [← sq'.w₁]
        /-
          C : Type u
          inst✝⁵ : CategoryTheory.Category.{v, u} C
          W : CategoryTheory.MorphismProperty C
          J : Type w
          inst✝⁴ : LinearOrder J
          inst✝³ : OrderBot J
          F : CategoryTheory.Functor J C
          c : CategoryTheory.Limits.Cocone F
          hc : CategoryTheory.Limits.IsColimit c
          X Y : C
          p : Quiver.Hom X Y
          f : Quiver.Hom (F.obj Bot.bot) X
          g : Quiver.Hom c.pt Y
          inst✝² : F.IsWellOrderContinuous
          inst✝¹ : SuccOrder J
          inst✝ : WellFoundedLT J
          hF : ∀ (j : J), Not (IsMax j) → CategoryTheory.HasLiftingProperty (F.map (Cate …
          j : J
          hj : Not (IsMax j)
          sq' : (CategoryTheory.HasLiftingProperty.transfiniteComposition.sqFunctor c p  …
          this : CategoryTheory.HasLiftingProperty (F.map (CategoryTheory.homOfLE ⋯)) p
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.homOfLE ⋯)) ⋯. …
        -/
        conv_rhs => rw [← sq'.sq.fac_left, ← F.map_comp_assoc]
        /-
          C : Type u
          inst✝⁵ : CategoryTheory.Category.{v, u} C
          W : CategoryTheory.MorphismProperty C
          J : Type w
          inst✝⁴ : LinearOrder J
          inst✝³ : OrderBot J
          F : CategoryTheory.Functor J C
          c : CategoryTheory.Limits.Cocone F
          hc : CategoryTheory.Limits.IsColimit c
          X Y : C
          p : Quiver.Hom X Y
          f : Quiver.Hom (F.obj Bot.bot) X
          g : Quiver.Hom c.pt Y
          inst✝² : F.IsWellOrderContinuous
          inst✝¹ : SuccOrder J
          inst✝ : WellFoundedLT J
          hF : ∀ (j : J), Not (IsMax j) → CategoryTheory.HasLiftingProperty (F.map (Cate …
          j : J
          hj : Not (IsMax j)
          sq' : (CategoryTheory.HasLiftingProperty.transfiniteComposition.sqFunctor c p  …
          this : CategoryTheory.HasLiftingProperty (F.map (CategoryTheory.homOfLE ⋯)) p
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.homOfLE ⋯)) ⋯. …
        -/
        rfl }
        /-
          🎉 no goals
        -/
                          /-
                            C : Type u
                            inst✝⁵ : CategoryTheory.Category.{v, u} C
                            W : CategoryTheory.MorphismProperty C
                            J : Type w
                            inst✝⁴ : LinearOrder J
                            inst✝³ : OrderBot J
                            F : CategoryTheory.Functor J C
                            c : CategoryTheory.Limits.Cocone F
                            hc : CategoryTheory.Limits.IsColimit c
                            X Y : C
                            p : Quiver.Hom X Y
                            f : Quiver.Hom (F.obj Bot.bot) X
                            g : Quiver.Hom c.pt Y
                            inst✝² : F.IsWellOrderContinuous
                            inst✝¹ : SuccOrder J
                            inst✝ : WellFoundedLT J
                            hF : ∀ (j : J), Not (IsMax j) → CategoryTheory.HasLiftingProperty (F.map (Cate …
                            j : J
                            hj : Not (IsMax j)
                            sq' : (CategoryTheory.HasLiftingProperty.transfiniteComposition.sqFunctor c p  …
                            ⊢ Eq ((CategoryTheory.HasLiftingProperty.transfiniteComposition.sqFunctor c p  …
                          -/
  map_succ j hj sq' := by aesop_cat
                          /-
                            🎉 no goals
                          -/
  lift j hj s := lift hj s
  map_lift j hj s i hij := map_lift hj s hij


lemma hasLift : sq.HasLift := by
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : Type w
    inst✝⁴ : LinearOrder J
    inst✝³ : OrderBot J
    F : CategoryTheory.Functor J C
    c : CategoryTheory.Limits.Cocone F
    hc : CategoryTheory.Limits.IsColimit c
    X Y : C
    p : Quiver.Hom X Y
    f : Quiver.Hom (F.obj Bot.bot) X
    g : Quiver.Hom c.pt Y
    inst✝² : F.IsWellOrderContinuous
    inst✝¹ : SuccOrder J
    inst✝ : WellFoundedLT J
    hF : ∀ (j : J), Not (IsMax j) → CategoryTheory.HasLiftingProperty (F.map (Cate …
    sq : CategoryTheory.CommSq f (c.ι.app Bot.bot) p g
    ⊢ sq.HasLift
  -/
  obtain ⟨s, hs⟩ := (wellOrderInductionData c f g hF).surjective { w₂ := sq.w }
  /-
    case intro
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : Type w
    inst✝⁴ : LinearOrder J
    inst✝³ : OrderBot J
    F : CategoryTheory.Functor J C
    c : CategoryTheory.Limits.Cocone F
    hc : CategoryTheory.Limits.IsColimit c
    X Y : C
    p : Quiver.Hom X Y
    f : Quiver.Hom (F.obj Bot.bot) X
    g : Quiver.Hom c.pt Y
    inst✝² : F.IsWellOrderContinuous
    inst✝¹ : SuccOrder J
    inst✝ : WellFoundedLT J
    hF : ∀ (j : J), Not (IsMax j) → CategoryTheory.HasLiftingProperty (F.map (Cate …
    sq : CategoryTheory.CommSq f (c.ι.app Bot.bot) p g
    s : ↑(CategoryTheory.HasLiftingProperty.transfiniteComposition.sqFunctor c p f …
    hs : Eq (Function.comp (fun s => s { unop := Bot.bot }) Subtype.val s) { f' := …
    ⊢ sq.HasLift
  -/
  replace hs := congr_arg SqStruct.f' hs
  /-
    case intro
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : Type w
    inst✝⁴ : LinearOrder J
    inst✝³ : OrderBot J
    F : CategoryTheory.Functor J C
    c : CategoryTheory.Limits.Cocone F
    hc : CategoryTheory.Limits.IsColimit c
    X Y : C
    p : Quiver.Hom X Y
    f : Quiver.Hom (F.obj Bot.bot) X
    g : Quiver.Hom c.pt Y
    inst✝² : F.IsWellOrderContinuous
    inst✝¹ : SuccOrder J
    inst✝ : WellFoundedLT J
    hF : ∀ (j : J), Not (IsMax j) → CategoryTheory.HasLiftingProperty (F.map (Cate …
    sq : CategoryTheory.CommSq f (c.ι.app Bot.bot) p g
    s : ↑(CategoryTheory.HasLiftingProperty.transfiniteComposition.sqFunctor c p f …
    hs : Eq (Function.comp (fun s => s { unop := Bot.bot }) Subtype.val s).f' { f' …
    ⊢ sq.HasLift
  -/
  dsimp at hs
  let t : Cocone F := Cocone.mk X
    { app j := (s.1 ⟨j⟩).f'
      naturality j j' g := by simpa using congr_arg SqStruct.f' (s.2 g.op) }
  /-
    case intro
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : Type w
    inst✝⁴ : LinearOrder J
    inst✝³ : OrderBot J
    F : CategoryTheory.Functor J C
    c : CategoryTheory.Limits.Cocone F
    hc : CategoryTheory.Limits.IsColimit c
    X Y : C
    p : Quiver.Hom X Y
    f : Quiver.Hom (F.obj Bot.bot) X
    g : Quiver.Hom c.pt Y
    inst✝² : F.IsWellOrderContinuous
    inst✝¹ : SuccOrder J
    inst✝ : WellFoundedLT J
    hF : ∀ (j : J), Not (IsMax j) → CategoryTheory.HasLiftingProperty (F.map (Cate …
    sq : CategoryTheory.CommSq f (c.ι.app Bot.bot) p g
    s : ↑(CategoryTheory.HasLiftingProperty.transfiniteComposition.sqFunctor c p f …
    hs : Eq (↑s { unop := Bot.bot }).f' f
    t : CategoryTheory.Limits.Cocone F := { pt := X, ι := { app := fun j => (↑s {  …
    ⊢ sq.HasLift
  -/
  let l := hc.desc t
  /-
    case intro
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : Type w
    inst✝⁴ : LinearOrder J
    inst✝³ : OrderBot J
    F : CategoryTheory.Functor J C
    c : CategoryTheory.Limits.Cocone F
    hc : CategoryTheory.Limits.IsColimit c
    X Y : C
    p : Quiver.Hom X Y
    f : Quiver.Hom (F.obj Bot.bot) X
    g : Quiver.Hom c.pt Y
    inst✝² : F.IsWellOrderContinuous
    inst✝¹ : SuccOrder J
    inst✝ : WellFoundedLT J
    hF : ∀ (j : J), Not (IsMax j) → CategoryTheory.HasLiftingProperty (F.map (Cate …
    sq : CategoryTheory.CommSq f (c.ι.app Bot.bot) p g
    s : ↑(CategoryTheory.HasLiftingProperty.transfiniteComposition.sqFunctor c p f …
    hs : Eq (↑s { unop := Bot.bot }).f' f
    t : CategoryTheory.Limits.Cocone F := { pt := X, ι := { app := fun j => (↑s {  …
    l : Quiver.Hom c.pt t.pt := hc.desc t
    ⊢ sq.HasLift
  -/
  have hl (j : J) : c.ι.app j ≫ l = (s.1 ⟨j⟩).f' := hc.fac t j
  exact ⟨⟨{
    l := l
    fac_left := by rw [hl, hs]
    fac_right := hc.hom_ext (fun j ↦ by rw [reassoc_of% (hl j), SqStruct.w₂])}⟩⟩


lemma hasLiftingProperty_ι_app_bot : HasLiftingProperty (c.ι.app ⊥) p where
  sq_hasLift sq := hasLift hc hF sq


