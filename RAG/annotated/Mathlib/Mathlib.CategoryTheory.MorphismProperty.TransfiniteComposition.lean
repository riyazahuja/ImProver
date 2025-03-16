/-- Given a functor `F : J ⥤ C` and `m : J`, this is the induced
functor `Set.Iio j ⥤ C`. -/
@[simps!]
def restrictionLT (F : J ⥤ C) (j : J) : Set.Iio j ⥤ C :=
  Monotone.functor (f := fun k ↦ k.1) (fun _ _ ↦ id) ⋙ F


/-- Given a functor `F : J ⥤ C` and `m : J`, this is the cocone with point `F.obj m`
for the restriction of `F` to `Set.Iio m`. -/
@[simps]
def coconeLT (F : J ⥤ C) (m : J) :
    Cocone (F.restrictionLT m) where
  pt := F.obj m
  ι :=
    { app := fun ⟨i, hi⟩ ↦ F.map (homOfLE hi.le)
      naturality := fun ⟨i₁, hi₁⟩ ⟨i₂, hi₂⟩ f ↦ by
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          J : Type w
          inst✝ : Preorder J
          F : CategoryTheory.Functor J C
          m : J
          x✝¹ x✝ : ↑(Set.Iio m)
          i₁ : J
          hi₁ : Membership.mem (Set.Iio m) i₁
          i₂ : J
          hi₂ : Membership.mem (Set.Iio m) i₂
          f : Quiver.Hom ⟨i₁, hi₁⟩ ⟨i₂, hi₂⟩
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.restrictionLT m).map f) ((fun x = …
        -/
        dsimp
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          J : Type w
          inst✝ : Preorder J
          F : CategoryTheory.Functor J C
          m : J
          x✝¹ x✝ : ↑(Set.Iio m)
          i₁ : J
          hi₁ : Membership.mem (Set.Iio m) i₁
          i₂ : J
          hi₂ : Membership.mem (Set.Iio m) i₂
          f : Quiver.Hom ⟨i₁, hi₁⟩ ⟨i₂, hi₂⟩
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map ((Monotone.functor ⋯).map f))  …
        -/
        rw [← F.map_comp, comp_id]
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          J : Type w
          inst✝ : Preorder J
          F : CategoryTheory.Functor J C
          m : J
          x✝¹ x✝ : ↑(Set.Iio m)
          i₁ : J
          hi₁ : Membership.mem (Set.Iio m) i₁
          i₂ : J
          hi₂ : Membership.mem (Set.Iio m) i₂
          f : Quiver.Hom ⟨i₁, hi₁⟩ ⟨i₂, hi₂⟩
          ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp ((Monotone.functor ⋯).map f) ( …
        -/
        rfl }
        /-
          🎉 no goals
        -/


/-- A functor `F : J ⥤ C` is well-order-continuous if for any limit element `m : J`,
`F.obj m` identifies to the colimit of the `F.obj j` for `j < m`. -/
class IsWellOrderContinuous (F : J ⥤ C) : Prop where
  nonempty_isColimit (m : J) (hm : Order.IsSuccLimit m) :
    Nonempty (IsColimit (F.coconeLT m))


/-- If `F : J ⥤ C` is well-order-continuous and `m : J` is a limit element, then
the cocone `F.coconeLT m` is colimit, i.e. `F.obj m` identifies to the colimit
of the `F.obj j` for `j < m`. -/
noncomputable def isColimitOfIsWellOrderContinuous (F : J ⥤ C) [F.IsWellOrderContinuous]
    (m : J) (hm : Order.IsSuccLimit m) :
    IsColimit (F.coconeLT m) := (IsWellOrderContinuous.nonempty_isColimit m hm).some


instance (F : ℕ ⥤ C) : F.IsWellOrderContinuous where
                                /-
                                  C : Type u
                                  inst✝¹ : CategoryTheory.Category.{v, u} C
                                  J : Type w
                                  inst✝ : Preorder J
                                  F : CategoryTheory.Functor Nat C
                                  m : Nat
                                  hm : Order.IsSuccLimit m
                                  ⊢ Nonempty (CategoryTheory.Limits.IsColimit (F.coconeLT m))
                                -/
  nonempty_isColimit m hm := by simp at hm
                                /-
                                  🎉 no goals
                                -/


lemma isWellOrderContinuous_of_iso {F G : J ⥤ C} (e : F ≅ G) [F.IsWellOrderContinuous] :
    G.IsWellOrderContinuous where
  nonempty_isColimit (m : J) (hm : Order.IsSuccLimit m) :=
    ⟨(IsColimit.precomposeHomEquiv (isoWhiskerLeft _ e) _).1
      (IsColimit.ofIsoColimit (F.isColimitOfIsWellOrderContinuous m hm)
         /-
           C : Type u
           inst✝² : CategoryTheory.Category.{v, u} C
           J : Type w
           inst✝¹ : Preorder J
           F G : CategoryTheory.Functor J C
           e : CategoryTheory.Iso F G
           inst✝ : F.IsWellOrderContinuous
           m : J
           hm : Order.IsSuccLimit m
           ⊢ ∀ (j : ↑(Set.Iio m)), Eq (CategoryTheory.CategoryStruct.comp ((F.coconeLT m) …
         -/
        (Cocones.ext (e.app _)))⟩
         /-
           🎉 no goals
         -/


/-- Given `W : MorphismProperty C` and a well-ordered type `J`, we say
that a morphism in `C` is a transfinite composition of morphisms in `W`
of shape `J` if it is of the form `c.ι.app ⊥ : F.obj ⊥ ⟶ c.pt`
where `c` is a colimit cocone for a well-order-continuous functor
`F : J ⥤ C` such that for any non-maximal `j : J`, the map
`F.map j ⟶ F.map (Order.succ j)` is in `W`. -/
inductive transfiniteCompositionsOfShape [WellFoundedLT J] : MorphismProperty C
  | mk (F : J ⥤ C) [F.IsWellOrderContinuous]
    (hF : ∀ (j : J) (_ : ¬IsMax j), W (F.map (homOfLE (Order.le_succ j))))
    (c : Cocone F) (hc : IsColimit c) : transfiniteCompositionsOfShape (c.ι.app ⊥)


instance [W.RespectsIso] : RespectsIso (W.transfiniteCompositionsOfShape J) where
  precomp := by
    /-
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      W : CategoryTheory.MorphismProperty C
      J : Type w
      inst✝⁴ : LinearOrder J
      inst✝³ : SuccOrder J
      inst✝² : OrderBot J
      inst✝¹ : WellFoundedLT J
      inst✝ : W.RespectsIso
      ⊢ ∀ {X Y Z : C} (i : Quiver.Hom X Y), CategoryTheory.MorphismProperty.isomorph …
    -/
    rintro X' X Y i (_ : IsIso i) _ ⟨F, hF, c, hc⟩
    let F' := F.copyObj (fun j ↦ if j = ⊥ then X' else F.obj j)
      (fun j ↦ if hj : j = ⊥ then
          eqToIso (by rw [hj]) ≪≫ (asIso i).symm ≪≫ eqToIso (if_pos hj).symm
        else eqToIso (if_neg hj).symm)
    /-
      case mk
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      W : CategoryTheory.MorphismProperty C
      J : Type w
      inst✝⁵ : LinearOrder J
      inst✝⁴ : SuccOrder J
      inst✝³ : OrderBot J
      inst✝² : WellFoundedLT J
      inst✝¹ : W.RespectsIso
      X' X Y : C
      F : CategoryTheory.Functor J C
      inst✝ : F.IsWellOrderContinuous
      hF : ∀ (j : J), Not (IsMax j) → W (F.map (CategoryTheory.homOfLE ⋯))
      c : CategoryTheory.Limits.Cocone F
      hc : CategoryTheory.Limits.IsColimit c
      i : Quiver.Hom X' (F.obj Bot.bot)
      hi✝ : CategoryTheory.IsIso i
      F' : CategoryTheory.Functor J C := F.copyObj (fun j => ite (Eq j Bot.bot) X' ( …
      ⊢ W.transfiniteCompositionsOfShape J (CategoryTheory.CategoryStruct.comp i (c. …
    -/
    let e : F ≅ F' := F.isoCopyObj _ _
    /-
      case mk
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      W : CategoryTheory.MorphismProperty C
      J : Type w
      inst✝⁵ : LinearOrder J
      inst✝⁴ : SuccOrder J
      inst✝³ : OrderBot J
      inst✝² : WellFoundedLT J
      inst✝¹ : W.RespectsIso
      X' X Y : C
      F : CategoryTheory.Functor J C
      inst✝ : F.IsWellOrderContinuous
      hF : ∀ (j : J), Not (IsMax j) → W (F.map (CategoryTheory.homOfLE ⋯))
      c : CategoryTheory.Limits.Cocone F
      hc : CategoryTheory.Limits.IsColimit c
      i : Quiver.Hom X' (F.obj Bot.bot)
      hi✝ : CategoryTheory.IsIso i
      F' : CategoryTheory.Functor J C := F.copyObj (fun j => ite (Eq j Bot.bot) X' ( …
      e : CategoryTheory.Iso F F' := F.isoCopyObj (fun j => ite (Eq j Bot.bot) X' (F …
      ⊢ W.transfiniteCompositionsOfShape J (CategoryTheory.CategoryStruct.comp i (c. …
    -/
    have := Functor.isWellOrderContinuous_of_iso e
    /-
      case mk
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      W : CategoryTheory.MorphismProperty C
      J : Type w
      inst✝⁵ : LinearOrder J
      inst✝⁴ : SuccOrder J
      inst✝³ : OrderBot J
      inst✝² : WellFoundedLT J
      inst✝¹ : W.RespectsIso
      X' X Y : C
      F : CategoryTheory.Functor J C
      inst✝ : F.IsWellOrderContinuous
      hF : ∀ (j : J), Not (IsMax j) → W (F.map (CategoryTheory.homOfLE ⋯))
      c : CategoryTheory.Limits.Cocone F
      hc : CategoryTheory.Limits.IsColimit c
      i : Quiver.Hom X' (F.obj Bot.bot)
      hi✝ : CategoryTheory.IsIso i
      F' : CategoryTheory.Functor J C := F.copyObj (fun j => ite (Eq j Bot.bot) X' ( …
      e : CategoryTheory.Iso F F' := F.isoCopyObj (fun j => ite (Eq j Bot.bot) X' (F …
      this : F'.IsWellOrderContinuous
      ⊢ W.transfiniteCompositionsOfShape J (CategoryTheory.CategoryStruct.comp i (c. …
    -/
    let c' : Cocone F' := (Cocones.precompose e.inv).obj c
    have : W.transfiniteCompositionsOfShape J (c'.ι.app ⊥) := by
      constructor
      · intro j hj
        exact (arrow_mk_iso_iff _ (((Functor.mapArrowFunctor _ _).mapIso e).app
          (Arrow.mk (homOfLE (Order.le_succ j))))).1 (hF j hj)
      · exact (IsColimit.precomposeInvEquiv e c).2 hc
    /-
      case mk
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      W : CategoryTheory.MorphismProperty C
      J : Type w
      inst✝⁵ : LinearOrder J
      inst✝⁴ : SuccOrder J
      inst✝³ : OrderBot J
      inst✝² : WellFoundedLT J
      inst✝¹ : W.RespectsIso
      X' X Y : C
      F : CategoryTheory.Functor J C
      inst✝ : F.IsWellOrderContinuous
      hF : ∀ (j : J), Not (IsMax j) → W (F.map (CategoryTheory.homOfLE ⋯))
      c : CategoryTheory.Limits.Cocone F
      hc : CategoryTheory.Limits.IsColimit c
      i : Quiver.Hom X' (F.obj Bot.bot)
      hi✝ : CategoryTheory.IsIso i
      F' : CategoryTheory.Functor J C := F.copyObj (fun j => ite (Eq j Bot.bot) X' ( …
      e : CategoryTheory.Iso F F' := F.isoCopyObj (fun j => ite (Eq j Bot.bot) X' (F …
      this✝ : F'.IsWellOrderContinuous
      c' : CategoryTheory.Limits.Cocone F' := (CategoryTheory.Limits.Cocones.precomp …
      this : W.transfiniteCompositionsOfShape J (c'.ι.app Bot.bot)
      ⊢ W.transfiniteCompositionsOfShape J (CategoryTheory.CategoryStruct.comp i (c. …
    -/
    exact MorphismProperty.of_eq _ this (if_pos rfl) rfl (by simp [c', e])
    /-
      🎉 no goals
    -/
  postcomp := by
    /-
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      W : CategoryTheory.MorphismProperty C
      J : Type w
      inst✝⁴ : LinearOrder J
      inst✝³ : SuccOrder J
      inst✝² : OrderBot J
      inst✝¹ : WellFoundedLT J
      inst✝ : W.RespectsIso
      ⊢ ∀ {X Y Z : C} (i : Quiver.Hom Y Z), CategoryTheory.MorphismProperty.isomorph …
    -/
    rintro _ _ _ i (_ : IsIso i) _ ⟨F, hF, c, hc⟩
    exact ⟨_, hF, { ι := c.ι ≫ (Functor.const _).map i },
      IsColimit.ofIsoColimit hc (Cocones.ext (asIso i))⟩


/-- A class of morphisms `W : MorphismProperty C` is stable under transfinite compositions
of shape `J` if for any well-order-continuous functor `F : J ⥤ C` such that
`F.obj j ⟶ F.obj (Order.succ j)` is in `W`, then `F.obj ⊥ ⟶ c.pt` is in `W`
for any colimit cocone `c : Cocone F`. -/
class IsStableUnderTransfiniteCompositionOfShape : Prop where
  le : W.transfiniteCompositionsOfShape J ≤ W


lemma transfiniteCompositionsOfShape_le  :
    W.transfiniteCompositionsOfShape J ≤ W :=
  IsStableUnderTransfiniteCompositionOfShape.le


variable {J} in
lemma mem_of_transfinite_composition {F : J ⥤ C} [F.IsWellOrderContinuous]
    (hF : ∀ (j : J) (_ : ¬IsMax j), W (F.map (homOfLE (Order.le_succ j))))
    {c : Cocone F} (hc : IsColimit c) : W (c.ι.app ⊥) :=
                                              /-
                                                C : Type u
                                                inst✝⁶ : CategoryTheory.Category.{v, u} C
                                                W : CategoryTheory.MorphismProperty C
                                                J : Type w
                                                inst✝⁵ : LinearOrder J
                                                inst✝⁴ : SuccOrder J
                                                inst✝³ : OrderBot J
                                                inst✝² : WellFoundedLT J
                                                inst✝¹ : W.IsStableUnderTransfiniteCompositionOfShape J
                                                F : CategoryTheory.Functor J C
                                                inst✝ : F.IsWellOrderContinuous
                                                hF : ∀ (j : J), Not (IsMax j) → W (F.map (CategoryTheory.homOfLE ⋯))
                                                c : CategoryTheory.Limits.Cocone F
                                                hc : CategoryTheory.Limits.IsColimit c
                                                ⊢ W.transfiniteCompositionsOfShape J (c.ι.app Bot.bot)
                                              -/
                                                              /-
                                                                🎉 no goals
                                                              -/
  W.transfiniteCompositionsOfShape_le J _ (by constructor <;> assumption)
                                                              /-
                                                                🎉 no goals
                                                              -/


/-- A class of morphisms `W : MorphismProperty C` is stable under infinite composition
if for any functor `F : ℕ ⥤ C` such that `F.obj n ⟶ F.obj (n + 1)` is in `W` for any `n : ℕ`,
the map `F.obj 0 ⟶ c.pt` is in `W` for any colimit cocone `c : Cocone F`. -/
abbrev IsStableUnderInfiniteComposition : Prop :=
  W.IsStableUnderTransfiniteCompositionOfShape ℕ


/-- A class of morphisms `W : MorphismProperty C` is stable under transfinite composition
if it is multiplicative and stable under transfinite composition of any shape
(in a certain universe). -/
class IsStableUnderTransfiniteComposition extends W.IsMultiplicative : Prop where
  isStableUnderTransfiniteCompositionOfShape
    (J : Type w) [LinearOrder J] [SuccOrder J] [OrderBot J] [WellFoundedLT J] :
    W.IsStableUnderTransfiniteCompositionOfShape J


