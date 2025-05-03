attribute [simp] CommSq.mk


/-- The (not necessarily limiting) `PullbackCone h i` implicit in the statement
that we have `CommSq f g h i`.
-/
def cone (s : CommSq f g h i) : PullbackCone h i :=
  PullbackCone.mk _ _ s.w


/-- The (not necessarily limiting) `PushoutCocone f g` implicit in the statement
that we have `CommSq f g h i`.
-/
def cocone (s : CommSq f g h i) : PushoutCocone f g :=
  PushoutCocone.mk _ _ s.w


@[simp]
theorem cone_fst (s : CommSq f g h i) : s.cone.fst = f :=
  rfl


@[simp]
theorem cone_snd (s : CommSq f g h i) : s.cone.snd = g :=
  rfl


@[simp]
theorem cocone_inl (s : CommSq f g h i) : s.cocone.inl = h :=
  rfl


@[simp]
theorem cocone_inr (s : CommSq f g h i) : s.cocone.inr = i :=
  rfl


/-- The pushout cocone in the opposite category associated to the cone of
a commutative square identifies to the cocone of the flipped commutative square in
the opposite category -/
def coneOp (p : CommSq f g h i) : p.cone.op ≅ p.flip.op.cocone :=
                                     /-
                                       C : Type u₁
                                       inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                       W X Y Z : C
                                       f : Quiver.Hom W X
                                       g : Quiver.Hom W Y
                                       h : Quiver.Hom X Z
                                       i : Quiver.Hom Y Z
                                       p : CategoryTheory.CommSq f g h i
                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp p.cone.op.inl (CategoryTheory.Iso.ref …
                                     -/
                                     /-
                                       🎉 no goals
                                     -/
  PushoutCocone.ext (Iso.refl _) (by aesop_cat) (by aesop_cat)
                                                    /-
                                                      🎉 no goals
                                                    -/


/-- The pullback cone in the opposite category associated to the cocone of
a commutative square identifies to the cone of the flipped commutative square in
the opposite category -/
def coconeOp (p : CommSq f g h i) : p.cocone.op ≅ p.flip.op.cone :=
                                    /-
                                      C : Type u₁
                                      inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                      W X Y Z : C
                                      f : Quiver.Hom W X
                                      g : Quiver.Hom W Y
                                      h : Quiver.Hom X Z
                                      i : Quiver.Hom Y Z
                                      p : CategoryTheory.CommSq f g h i
                                      ⊢ Eq p.cocone.op.fst (CategoryTheory.CategoryStruct.comp (CategoryTheory.Iso.r …
                                    -/
                                    /-
                                      🎉 no goals
                                    -/
  PullbackCone.ext (Iso.refl _) (by aesop_cat) (by aesop_cat)
                                                   /-
                                                     🎉 no goals
                                                   -/


/-- The pushout cocone obtained from the pullback cone associated to a
commutative square in the opposite category identifies to the cocone associated
to the flipped square. -/
def coneUnop {W X Y Z : Cᵒᵖ} {f : W ⟶ X} {g : W ⟶ Y} {h : X ⟶ Z} {i : Y ⟶ Z} (p : CommSq f g h i) :
    p.cone.unop ≅ p.flip.unop.cocone :=
                                     /-
                                       C : Type u₁
                                       inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                       W✝ X✝ Y✝ Z✝ : C
                                       f✝ : Quiver.Hom W✝ X✝
                                       g✝ : Quiver.Hom W✝ Y✝
                                       h✝ : Quiver.Hom X✝ Z✝
                                       i✝ : Quiver.Hom Y✝ Z✝
                                       W X Y Z : Opposite C
                                       f : Quiver.Hom W X
                                       g : Quiver.Hom W Y
                                       h : Quiver.Hom X Z
                                       i : Quiver.Hom Y Z
                                       p : CategoryTheory.CommSq f g h i
                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp p.cone.unop.inl (CategoryTheory.Iso.r …
                                     -/
                                     /-
                                       🎉 no goals
                                     -/
  PushoutCocone.ext (Iso.refl _) (by aesop_cat) (by aesop_cat)
                                                    /-
                                                      🎉 no goals
                                                    -/


/-- The pullback cone obtained from the pushout cone associated to a
commutative square in the opposite category identifies to the cone associated
to the flipped square. -/
def coconeUnop {W X Y Z : Cᵒᵖ} {f : W ⟶ X} {g : W ⟶ Y} {h : X ⟶ Z} {i : Y ⟶ Z}
    (p : CommSq f g h i) : p.cocone.unop ≅ p.flip.unop.cone :=
                                    /-
                                      C : Type u₁
                                      inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                      W✝ X✝ Y✝ Z✝ : C
                                      f✝ : Quiver.Hom W✝ X✝
                                      g✝ : Quiver.Hom W✝ Y✝
                                      h✝ : Quiver.Hom X✝ Z✝
                                      i✝ : Quiver.Hom Y✝ Z✝
                                      W X Y Z : Opposite C
                                      f : Quiver.Hom W X
                                      g : Quiver.Hom W Y
                                      h : Quiver.Hom X Z
                                      i : Quiver.Hom Y Z
                                      p : CategoryTheory.CommSq f g h i
                                      ⊢ Eq p.cocone.unop.fst (CategoryTheory.CategoryStruct.comp (CategoryTheory.Iso …
                                    -/
                                    /-
                                      🎉 no goals
                                    -/
  PullbackCone.ext (Iso.refl _) (by aesop_cat) (by aesop_cat)
                                                   /-
                                                     🎉 no goals
                                                   -/


/-- The proposition that a square
```
  P --fst--> X
  |          |
 snd         f
  |          |
  v          v
  Y ---g---> Z

```
is a pullback square. (Also known as a fibered product or cartesian square.)
-/
structure IsPullback {P X Y Z : C} (fst : P ⟶ X) (snd : P ⟶ Y) (f : X ⟶ Z) (g : Y ⟶ Z) extends
  CommSq fst snd f g : Prop where
  /-- the pullback cone is a limit -/
  isLimit' : Nonempty (IsLimit (PullbackCone.mk _ _ w))


/-- The proposition that a square
```
  Z ---f---> X
  |          |
  g         inl
  |          |
  v          v
  Y --inr--> P

```
is a pushout square. (Also known as a fiber coproduct or cocartesian square.)
-/
structure IsPushout {Z X Y P : C} (f : Z ⟶ X) (g : Z ⟶ Y) (inl : X ⟶ P) (inr : Y ⟶ P) extends
  CommSq f g inl inr : Prop where
  /-- the pushout cocone is a colimit -/
  isColimit' : Nonempty (IsColimit (PushoutCocone.mk _ _ w))


/-- A *bicartesian* square is a commutative square
```
  W ---f---> X
  |          |
  g          h
  |          |
  v          v
  Y ---i---> Z

```
that is both a pullback square and a pushout square.
-/
structure BicartesianSq {W X Y Z : C} (f : W ⟶ X) (g : W ⟶ Y) (h : X ⟶ Z) (i : Y ⟶ Z) extends
  IsPullback f g h i, IsPushout f g h i : Prop

-- Lean should make these parent projections as `lemma`, not `def`.

/-- The (limiting) `PullbackCone f g` implicit in the statement
that we have an `IsPullback fst snd f g`.
-/
def cone (h : IsPullback fst snd f g) : PullbackCone f g :=
  h.toCommSq.cone


@[simp]
theorem cone_fst (h : IsPullback fst snd f g) : h.cone.fst = fst :=
  rfl


@[simp]
theorem cone_snd (h : IsPullback fst snd f g) : h.cone.snd = snd :=
  rfl


/-- The cone obtained from `IsPullback fst snd f g` is a limit cone.
-/
noncomputable def isLimit (h : IsPullback fst snd f g) : IsLimit h.cone :=
  h.isLimit'.some


/-- API for PullbackCone.IsLimit.lift for `IsPullback` -/
noncomputable def lift (hP : IsPullback fst snd f g) {W : C} (h : W ⟶ X) (k : W ⟶ Y)
    (w : h ≫ f = k ≫ g) : W ⟶ P :=
  PullbackCone.IsLimit.lift hP.isLimit h k w


@[reassoc (attr := simp)]
lemma lift_fst (hP : IsPullback fst snd f g) {W : C} (h : W ⟶ X) (k : W ⟶ Y)
    (w : h ≫ f = k ≫ g) : hP.lift h k w ≫ fst = h :=
  PullbackCone.IsLimit.lift_fst hP.isLimit h k w


@[reassoc (attr := simp)]
lemma lift_snd (hP : IsPullback fst snd f g) {W : C} (h : W ⟶ X) (k : W ⟶ Y)
    (w : h ≫ f = k ≫ g) : hP.lift h k w ≫ snd = k :=
  PullbackCone.IsLimit.lift_snd hP.isLimit h k w


lemma hom_ext (hP : IsPullback fst snd f g) {W : C} {k l : W ⟶ P}
    (h₀ : k ≫ fst = l ≫ fst) (h₁ : k ≫ snd = l ≫ snd) : k = l :=
  PullbackCone.IsLimit.hom_ext hP.isLimit h₀ h₁


/-- If `c` is a limiting pullback cone, then we have an `IsPullback c.fst c.snd f g`. -/
theorem of_isLimit {c : PullbackCone f g} (h : Limits.IsLimit c) : IsPullback c.fst c.snd f g :=
  { w := c.condition
    isLimit' := ⟨IsLimit.ofIsoLimit h (Limits.PullbackCone.ext (Iso.refl _)
          /-
            C : Type u₁
            inst✝ : CategoryTheory.Category.{v₁, u₁} C
            X Y Z : C
            f : Quiver.Hom X Z
            g : Quiver.Hom Y Z
            c : CategoryTheory.Limits.PullbackCone f g
            h : CategoryTheory.Limits.IsLimit c
            ⊢ Eq c.fst (CategoryTheory.CategoryStruct.comp (CategoryTheory.Iso.refl c.pt). …
          -/
          /-
            🎉 no goals
          -/
      (by aesop_cat) (by aesop_cat))⟩ }
                         /-
                           🎉 no goals
                         -/


/-- A variant of `of_isLimit` that is more useful with `apply`. -/
theorem of_isLimit' (w : CommSq fst snd f g) (h : Limits.IsLimit w.cone) :
    IsPullback fst snd f g :=
  of_isLimit h


/-- Variant of `of_isLimit` for an arbitrary cone on a diagram `WalkingCospan ⥤ C`. -/
lemma of_isLimit_cone {D : WalkingCospan ⥤ C} {c : Cone D} (hc : IsLimit c) :
    IsPullback (c.π.app .left) (c.π.app .right) (D.map WalkingCospan.Hom.inl)
      (D.map WalkingCospan.Hom.inr) where
          /-
            C : Type u₁
            inst✝ : CategoryTheory.Category.{v₁, u₁} C
            D : CategoryTheory.Functor CategoryTheory.Limits.WalkingCospan C
            c : CategoryTheory.Limits.Cone D
            hc : CategoryTheory.Limits.IsLimit c
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.π.app CategoryTheory.Limits.Walkin …
          -/
  w := by simp_rw [Cone.w]
          /-
            🎉 no goals
          -/
  isLimit' := ⟨IsLimit.equivOfNatIsoOfIso _ _ _ (PullbackCone.isoMk c) hc⟩


lemma hasPullback (h : IsPullback fst snd f g) : HasPullback f g where
  exists_limit := ⟨⟨h.cone, h.isLimit⟩⟩


/-- The pullback provided by `HasPullback f g` fits into an `IsPullback`. -/
theorem of_hasPullback (f : X ⟶ Z) (g : Y ⟶ Z) [HasPullback f g] :
    IsPullback (pullback.fst f g) (pullback.snd f g) f g :=
  of_isLimit (limit.isLimit (cospan f g))


/-- If `c` is a limiting binary product cone, and we have a terminal object,
then we have `IsPullback c.fst c.snd 0 0`
(where each `0` is the unique morphism to the terminal object). -/
theorem of_is_product {c : BinaryFan X Y} (h : Limits.IsLimit c) (t : IsTerminal Z) :
    IsPullback c.fst c.snd (t.from _) (t.from _) :=
  of_isLimit
    (isPullbackOfIsTerminalIsProduct _ _ _ _ t
      (IsLimit.ofIsoLimit h
        (Limits.Cones.ext (Iso.refl c.pt)
          (by
            /-
              C : Type u₁
              inst✝ : CategoryTheory.Category.{v₁, u₁} C
              X Y Z : C
              c : CategoryTheory.Limits.BinaryFan X Y
              h : CategoryTheory.Limits.IsLimit c
              t : CategoryTheory.Limits.IsTerminal Z
              ⊢ ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (c.π.a …
            -/
            rintro ⟨⟨⟩⟩ <;>
                /-
                  case mk.left
                  C : Type u₁
                  inst✝ : CategoryTheory.Category.{v₁, u₁} C
                  X Y Z : C
                  c : CategoryTheory.Limits.BinaryFan X Y
                  h : CategoryTheory.Limits.IsLimit c
                  t : CategoryTheory.Limits.IsTerminal Z
                  ⊢ Eq (c.π.app { as := CategoryTheory.Limits.WalkingPair.left }) (CategoryTheor …
                -/
                /-
                  case mk.left
                  C : Type u₁
                  inst✝ : CategoryTheory.Category.{v₁, u₁} C
                  X Y Z : C
                  c : CategoryTheory.Limits.BinaryFan X Y
                  h : CategoryTheory.Limits.IsLimit c
                  t : CategoryTheory.Limits.IsTerminal Z
                  ⊢ Eq c.fst (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct. …
                -/
                /-
                  🎉 no goals
                -/
                /-
                  case mk.right
                  C : Type u₁
                  inst✝ : CategoryTheory.Category.{v₁, u₁} C
                  X Y Z : C
                  c : CategoryTheory.Limits.BinaryFan X Y
                  h : CategoryTheory.Limits.IsLimit c
                  t : CategoryTheory.Limits.IsTerminal Z
                  ⊢ Eq c.snd (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct. …
                -/
                simp))))
                /-
                  🎉 no goals
                -/


/-- A variant of `of_is_product` that is more useful with `apply`. -/
theorem of_is_product' (h : Limits.IsLimit (BinaryFan.mk fst snd)) (t : IsTerminal Z) :
    IsPullback fst snd (t.from _) (t.from _) :=
  of_is_product h t


theorem of_hasBinaryProduct' [HasBinaryProduct X Y] [HasTerminal C] :
    IsPullback Limits.prod.fst Limits.prod.snd (terminal.from X) (terminal.from Y) :=
  of_is_product (limit.isLimit _) terminalIsTerminal


theorem of_hasBinaryProduct [HasBinaryProduct X Y] [HasZeroObject C] [HasZeroMorphisms C] :
    IsPullback Limits.prod.fst Limits.prod.snd (0 : X ⟶ 0) (0 : Y ⟶ 0) := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    inst✝² : CategoryTheory.Limits.HasBinaryProduct X Y
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    ⊢ CategoryTheory.IsPullback CategoryTheory.Limits.prod.fst CategoryTheory.Limi …
  -/
  convert @of_is_product _ _ X Y 0 _ (limit.isLimit _) HasZeroObject.zeroIsTerminal
        /-
          case h.e'_9.h
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          X Y : C
          inst✝² : CategoryTheory.Limits.HasBinaryProduct X Y
          inst✝¹ : CategoryTheory.Limits.HasZeroObject C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          e_4✝ : Eq X ((CategoryTheory.Limits.pair X Y).obj { as := CategoryTheory.Limit …
          ⊢ Eq 0 (CategoryTheory.Limits.HasZeroObject.zeroIsTerminal.from ((CategoryTheo …
        -/
        /-
          🎉 no goals
        -/
    <;> subsingleton
        /-
          🎉 no goals
        -/


/-- Any object at the top left of a pullback square is isomorphic to the object at the top left
of any other pullback square with the same cospan. -/
noncomputable def isoIsPullback (h : IsPullback fst snd f g) (h' : IsPullback fst' snd' f g) :
    P ≅ P' :=
  IsLimit.conePointUniqueUpToIso h.isLimit h'.isLimit


@[reassoc (attr := simp)]
theorem isoIsPullback_hom_fst (h : IsPullback fst snd f g) (h' : IsPullback fst' snd' f g) :
    (h.isoIsPullback _ _ h').hom ≫ fst' = fst :=
  IsLimit.conePointUniqueUpToIso_hom_comp h.isLimit h'.isLimit WalkingCospan.left


@[reassoc (attr := simp)]
theorem isoIsPullback_hom_snd (h : IsPullback fst snd f g) (h' : IsPullback fst' snd' f g) :
    (h.isoIsPullback _ _ h').hom ≫ snd' = snd :=
  IsLimit.conePointUniqueUpToIso_hom_comp h.isLimit h'.isLimit WalkingCospan.right


@[reassoc (attr := simp)]
theorem isoIsPullback_inv_fst (h : IsPullback fst snd f g) (h' : IsPullback fst' snd' f g) :
    (h.isoIsPullback _ _ h').inv ≫ fst = fst' := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P X Y Z : C
    fst : Quiver.Hom P X
    snd : Quiver.Hom P Y
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    P' : C
    fst' : Quiver.Hom P' X
    snd' : Quiver.Hom P' Y
    h : CategoryTheory.IsPullback fst snd f g
    h' : CategoryTheory.IsPullback fst' snd' f g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsPullback.isoIsPullb …
  -/
  simp only [Iso.inv_comp_eq, isoIsPullback_hom_fst]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem isoIsPullback_inv_snd (h : IsPullback fst snd f g) (h' : IsPullback fst' snd' f g) :
    (h.isoIsPullback _ _ h').inv ≫ snd = snd' := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P X Y Z : C
    fst : Quiver.Hom P X
    snd : Quiver.Hom P Y
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    P' : C
    fst' : Quiver.Hom P' X
    snd' : Quiver.Hom P' Y
    h : CategoryTheory.IsPullback fst snd f g
    h' : CategoryTheory.IsPullback fst' snd' f g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsPullback.isoIsPullb …
  -/
  simp only [Iso.inv_comp_eq, isoIsPullback_hom_snd]
  /-
    🎉 no goals
  -/


/-- Any object at the top left of a pullback square is
isomorphic to the pullback provided by the `HasLimit` API. -/
noncomputable def isoPullback (h : IsPullback fst snd f g) [HasPullback f g] : P ≅ pullback f g :=
  (limit.isoLimitCone ⟨_, h.isLimit⟩).symm


@[reassoc (attr := simp)]
theorem isoPullback_hom_fst (h : IsPullback fst snd f g) [HasPullback f g] :
    h.isoPullback.hom ≫ pullback.fst _ _ = fst := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    P X Y Z : C
    fst : Quiver.Hom P X
    snd : Quiver.Hom P Y
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    h : CategoryTheory.IsPullback fst snd f g
    inst✝ : CategoryTheory.Limits.HasPullback f g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp h.isoPullback.hom (CategoryTheory.Lim …
  -/
  dsimp [isoPullback, cone, CommSq.cone]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    P X Y Z : C
    fst : Quiver.Hom P X
    snd : Quiver.Hom P Y
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    h : CategoryTheory.IsPullback fst snd f g
    inst✝ : CategoryTheory.Limits.HasPullback f g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.isoLimit …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem isoPullback_hom_snd (h : IsPullback fst snd f g) [HasPullback f g] :
    h.isoPullback.hom ≫ pullback.snd _ _ = snd := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    P X Y Z : C
    fst : Quiver.Hom P X
    snd : Quiver.Hom P Y
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    h : CategoryTheory.IsPullback fst snd f g
    inst✝ : CategoryTheory.Limits.HasPullback f g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp h.isoPullback.hom (CategoryTheory.Lim …
  -/
  dsimp [isoPullback, cone, CommSq.cone]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    P X Y Z : C
    fst : Quiver.Hom P X
    snd : Quiver.Hom P Y
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    h : CategoryTheory.IsPullback fst snd f g
    inst✝ : CategoryTheory.Limits.HasPullback f g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.isoLimit …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem isoPullback_inv_fst (h : IsPullback fst snd f g) [HasPullback f g] :
                                                     /-
                                                       C : Type u₁
                                                       inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                       P X Y Z : C
                                                       fst : Quiver.Hom P X
                                                       snd : Quiver.Hom P Y
                                                       f : Quiver.Hom X Z
                                                       g : Quiver.Hom Y Z
                                                       h : CategoryTheory.IsPullback fst snd f g
                                                       inst✝ : CategoryTheory.Limits.HasPullback f g
                                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp h.isoPullback.inv fst) (CategoryTheor …
                                                     -/
    h.isoPullback.inv ≫ fst = pullback.fst _ _ := by simp [Iso.inv_comp_eq]
                                                     /-
                                                       🎉 no goals
                                                     -/


@[reassoc (attr := simp)]
theorem isoPullback_inv_snd (h : IsPullback fst snd f g) [HasPullback f g] :
                                                     /-
                                                       C : Type u₁
                                                       inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                       P X Y Z : C
                                                       fst : Quiver.Hom P X
                                                       snd : Quiver.Hom P Y
                                                       f : Quiver.Hom X Z
                                                       g : Quiver.Hom Y Z
                                                       h : CategoryTheory.IsPullback fst snd f g
                                                       inst✝ : CategoryTheory.Limits.HasPullback f g
                                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp h.isoPullback.inv snd) (CategoryTheor …
                                                     -/
    h.isoPullback.inv ≫ snd = pullback.snd _ _ := by simp [Iso.inv_comp_eq]
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem of_iso_pullback (h : CommSq fst snd f g) [HasPullback f g] (i : P ≅ pullback f g)
    (w₁ : i.hom ≫ pullback.fst _ _ = fst) (w₂ : i.hom ≫ pullback.snd _ _ = snd) :
      IsPullback fst snd f g :=
  of_isLimit' h
    (Limits.IsLimit.ofIsoLimit (limit.isLimit _)
      (@PullbackCone.ext _ _ _ _ _ _ _ (PullbackCone.mk _ _ _) _ i w₁.symm w₂.symm).symm)


theorem of_horiz_isIso [IsIso fst] [IsIso g] (sq : CommSq fst snd f g) : IsPullback fst snd f g :=
  of_isLimit' sq
    (by
      refine
        PullbackCone.IsLimit.mk _ (fun s => s.fst ≫ inv fst) (by aesop_cat)
          (fun s => ?_) (by aesop_cat)
      /-
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        P X Y Z : C
        fst : Quiver.Hom P X
        snd : Quiver.Hom P Y
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        inst✝¹ : CategoryTheory.IsIso fst
        inst✝ : CategoryTheory.IsIso g
        sq : CategoryTheory.CommSq fst snd f g
        s : CategoryTheory.Limits.PullbackCone f g
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s => CategoryTheory.CategoryStr …
      -/
      simp only [← cancel_mono g, Category.assoc, ← sq.w, IsIso.inv_hom_id_assoc, s.condition])
      /-
        🎉 no goals
      -/


lemma of_iso (h : IsPullback fst snd f g)
    {P' X' Y' Z' : C} {fst' : P' ⟶ X'} {snd' : P' ⟶ Y'} {f' : X' ⟶ Z'} {g' : Y' ⟶ Z'}
    (e₁ : P ≅ P') (e₂ : X ≅ X') (e₃ : Y ≅ Y') (e₄ : Z ≅ Z')
    (commfst : fst ≫ e₂.hom = e₁.hom ≫ fst')
    (commsnd : snd ≫ e₃.hom = e₁.hom ≫ snd')
    (commf : f ≫ e₄.hom = e₂.hom ≫ f')
    (commg : g ≫ e₄.hom = e₃.hom ≫ g') :
    IsPullback fst' snd' f' g' where
  w := by
    rw [← cancel_epi e₁.hom, ← reassoc_of% commfst, ← commf,
      ← reassoc_of% commsnd, ← commg, h.w_assoc]
  isLimit' :=
    ⟨(IsLimit.postcomposeInvEquiv
        (cospanExt e₂ e₃ e₄ commf.symm commg.symm) _).1
          (IsLimit.ofIsoLimit h.isLimit (by
            /-
              C : Type u₁
              inst✝ : CategoryTheory.Category.{v₁, u₁} C
              P X Y Z : C
              fst : Quiver.Hom P X
              snd : Quiver.Hom P Y
              f : Quiver.Hom X Z
              g : Quiver.Hom Y Z
              h : CategoryTheory.IsPullback fst snd f g
              P' X' Y' Z' : C
              fst' : Quiver.Hom P' X'
              snd' : Quiver.Hom P' Y'
              f' : Quiver.Hom X' Z'
              g' : Quiver.Hom Y' Z'
              e₁ : CategoryTheory.Iso P P'
              e₂ : CategoryTheory.Iso X X'
              e₃ : CategoryTheory.Iso Y Y'
              e₄ : CategoryTheory.Iso Z Z'
              commfst : Eq (CategoryTheory.CategoryStruct.comp fst e₂.hom) (CategoryTheory.C …
              commsnd : Eq (CategoryTheory.CategoryStruct.comp snd e₃.hom) (CategoryTheory.C …
              commf : Eq (CategoryTheory.CategoryStruct.comp f e₄.hom) (CategoryTheory.Categ …
              commg : Eq (CategoryTheory.CategoryStruct.comp g e₄.hom) (CategoryTheory.Categ …
              ⊢ CategoryTheory.Iso h.cone ((CategoryTheory.Limits.Cones.postcompose (Categor …
            -/
            refine PullbackCone.ext e₁ ?_ ?_
              /-
                case refine_1
                C : Type u₁
                inst✝ : CategoryTheory.Category.{v₁, u₁} C
                P X Y Z : C
                fst : Quiver.Hom P X
                snd : Quiver.Hom P Y
                f : Quiver.Hom X Z
                g : Quiver.Hom Y Z
                h : CategoryTheory.IsPullback fst snd f g
                P' X' Y' Z' : C
                fst' : Quiver.Hom P' X'
                snd' : Quiver.Hom P' Y'
                f' : Quiver.Hom X' Z'
                g' : Quiver.Hom Y' Z'
                e₁ : CategoryTheory.Iso P P'
                e₂ : CategoryTheory.Iso X X'
                e₃ : CategoryTheory.Iso Y Y'
                e₄ : CategoryTheory.Iso Z Z'
                commfst : Eq (CategoryTheory.CategoryStruct.comp fst e₂.hom) (CategoryTheory.C …
                commsnd : Eq (CategoryTheory.CategoryStruct.comp snd e₃.hom) (CategoryTheory.C …
                commf : Eq (CategoryTheory.CategoryStruct.comp f e₄.hom) (CategoryTheory.Categ …
                commg : Eq (CategoryTheory.CategoryStruct.comp g e₄.hom) (CategoryTheory.Categ …
                ⊢ Eq h.cone.fst (CategoryTheory.CategoryStruct.comp e₁.hom (CategoryTheory.Lim …
              -/
            · change fst = e₁.hom ≫ fst' ≫ e₂.inv
              /-
                case refine_1
                C : Type u₁
                inst✝ : CategoryTheory.Category.{v₁, u₁} C
                P X Y Z : C
                fst : Quiver.Hom P X
                snd : Quiver.Hom P Y
                f : Quiver.Hom X Z
                g : Quiver.Hom Y Z
                h : CategoryTheory.IsPullback fst snd f g
                P' X' Y' Z' : C
                fst' : Quiver.Hom P' X'
                snd' : Quiver.Hom P' Y'
                f' : Quiver.Hom X' Z'
                g' : Quiver.Hom Y' Z'
                e₁ : CategoryTheory.Iso P P'
                e₂ : CategoryTheory.Iso X X'
                e₃ : CategoryTheory.Iso Y Y'
                e₄ : CategoryTheory.Iso Z Z'
                commfst : Eq (CategoryTheory.CategoryStruct.comp fst e₂.hom) (CategoryTheory.C …
                commsnd : Eq (CategoryTheory.CategoryStruct.comp snd e₃.hom) (CategoryTheory.C …
                commf : Eq (CategoryTheory.CategoryStruct.comp f e₄.hom) (CategoryTheory.Categ …
                commg : Eq (CategoryTheory.CategoryStruct.comp g e₄.hom) (CategoryTheory.Categ …
                ⊢ Eq fst (CategoryTheory.CategoryStruct.comp e₁.hom (CategoryTheory.CategorySt …
              -/
              rw [← reassoc_of% commfst, e₂.hom_inv_id, Category.comp_id]
              /-
                🎉 no goals
              -/
              /-
                case refine_2
                C : Type u₁
                inst✝ : CategoryTheory.Category.{v₁, u₁} C
                P X Y Z : C
                fst : Quiver.Hom P X
                snd : Quiver.Hom P Y
                f : Quiver.Hom X Z
                g : Quiver.Hom Y Z
                h : CategoryTheory.IsPullback fst snd f g
                P' X' Y' Z' : C
                fst' : Quiver.Hom P' X'
                snd' : Quiver.Hom P' Y'
                f' : Quiver.Hom X' Z'
                g' : Quiver.Hom Y' Z'
                e₁ : CategoryTheory.Iso P P'
                e₂ : CategoryTheory.Iso X X'
                e₃ : CategoryTheory.Iso Y Y'
                e₄ : CategoryTheory.Iso Z Z'
                commfst : Eq (CategoryTheory.CategoryStruct.comp fst e₂.hom) (CategoryTheory.C …
                commsnd : Eq (CategoryTheory.CategoryStruct.comp snd e₃.hom) (CategoryTheory.C …
                commf : Eq (CategoryTheory.CategoryStruct.comp f e₄.hom) (CategoryTheory.Categ …
                commg : Eq (CategoryTheory.CategoryStruct.comp g e₄.hom) (CategoryTheory.Categ …
                ⊢ Eq h.cone.snd (CategoryTheory.CategoryStruct.comp e₁.hom (CategoryTheory.Lim …
              -/
            · change snd = e₁.hom ≫ snd' ≫ e₃.inv
              /-
                case refine_2
                C : Type u₁
                inst✝ : CategoryTheory.Category.{v₁, u₁} C
                P X Y Z : C
                fst : Quiver.Hom P X
                snd : Quiver.Hom P Y
                f : Quiver.Hom X Z
                g : Quiver.Hom Y Z
                h : CategoryTheory.IsPullback fst snd f g
                P' X' Y' Z' : C
                fst' : Quiver.Hom P' X'
                snd' : Quiver.Hom P' Y'
                f' : Quiver.Hom X' Z'
                g' : Quiver.Hom Y' Z'
                e₁ : CategoryTheory.Iso P P'
                e₂ : CategoryTheory.Iso X X'
                e₃ : CategoryTheory.Iso Y Y'
                e₄ : CategoryTheory.Iso Z Z'
                commfst : Eq (CategoryTheory.CategoryStruct.comp fst e₂.hom) (CategoryTheory.C …
                commsnd : Eq (CategoryTheory.CategoryStruct.comp snd e₃.hom) (CategoryTheory.C …
                commf : Eq (CategoryTheory.CategoryStruct.comp f e₄.hom) (CategoryTheory.Categ …
                commg : Eq (CategoryTheory.CategoryStruct.comp g e₄.hom) (CategoryTheory.Categ …
                ⊢ Eq snd (CategoryTheory.CategoryStruct.comp e₁.hom (CategoryTheory.CategorySt …
              -/
              rw [← reassoc_of% commsnd, e₃.hom_inv_id, Category.comp_id]))⟩
              /-
                🎉 no goals
              -/

lemma isIso_fst_of_mono (h : IsPullback fst snd f f) : IsIso fst :=
  h.cone.isIso_fst_of_mono_of_isLimit h.isLimit


lemma isIso_snd_iso_of_mono {P X Y : C} {fst : P ⟶ X} {snd : P ⟶ X} {f : X ⟶ Y} [Mono f]
    (h : IsPullback fst snd f f) : IsIso snd :=
  h.cone.isIso_snd_of_mono_of_isLimit h.isLimit


/-- The (colimiting) `PushoutCocone f g` implicit in the statement
that we have an `IsPushout f g inl inr`.
-/
def cocone (h : IsPushout f g inl inr) : PushoutCocone f g :=
  h.toCommSq.cocone


@[simp]
theorem cocone_inl (h : IsPushout f g inl inr) : h.cocone.inl = inl :=
  rfl


@[simp]
theorem cocone_inr (h : IsPushout f g inl inr) : h.cocone.inr = inr :=
  rfl


/-- The cocone obtained from `IsPushout f g inl inr` is a colimit cocone.
-/
noncomputable def isColimit (h : IsPushout f g inl inr) : IsColimit h.cocone :=
  h.isColimit'.some


/-- API for PushoutCocone.IsColimit.lift for `IsPushout` -/
noncomputable def desc (hP : IsPushout f g inl inr) {W : C} (h : X ⟶ W) (k : Y ⟶ W)
    (w : f ≫ h = g ≫ k) : P ⟶ W :=
  PushoutCocone.IsColimit.desc hP.isColimit h k w


@[reassoc (attr := simp)]
lemma inl_desc (hP : IsPushout f g inl inr) {W : C} (h : X ⟶ W) (k : Y ⟶ W)
    (w : f ≫ h = g ≫ k) : inl ≫ hP.desc h k w = h :=
  PushoutCocone.IsColimit.inl_desc hP.isColimit h k w


@[reassoc (attr := simp)]
lemma inr_desc (hP : IsPushout f g inl inr) {W : C} (h : X ⟶ W) (k : Y ⟶ W)
    (w : f ≫ h = g ≫ k) : inr ≫ hP.desc h k w = k :=
  PushoutCocone.IsColimit.inr_desc hP.isColimit h k w


lemma hom_ext (hP : IsPushout f g inl inr) {W : C} {k l : P ⟶ W}
    (h₀ : inl ≫ k = inl ≫ l) (h₁ : inr ≫ k = inr ≫ l) : k = l :=
  PushoutCocone.IsColimit.hom_ext hP.isColimit h₀ h₁


/-- If `c` is a colimiting pushout cocone, then we have an `IsPushout f g c.inl c.inr`. -/
theorem of_isColimit {c : PushoutCocone f g} (h : Limits.IsColimit c) : IsPushout f g c.inl c.inr :=
  { w := c.condition
    isColimit' :=
      ⟨IsColimit.ofIsoColimit h (Limits.PushoutCocone.ext (Iso.refl _)
            /-
              C : Type u₁
              inst✝ : CategoryTheory.Category.{v₁, u₁} C
              Z X Y : C
              f : Quiver.Hom Z X
              g : Quiver.Hom Z Y
              c : CategoryTheory.Limits.PushoutCocone f g
              h : CategoryTheory.Limits.IsColimit c
              ⊢ Eq (CategoryTheory.CategoryStruct.comp c.inl (CategoryTheory.Iso.refl c.pt). …
            -/
            /-
              🎉 no goals
            -/
        (by aesop_cat) (by aesop_cat))⟩ }
                           /-
                             🎉 no goals
                           -/


/-- A variant of `of_isColimit` that is more useful with `apply`. -/
theorem of_isColimit' (w : CommSq f g inl inr) (h : Limits.IsColimit w.cocone) :
    IsPushout f g inl inr :=
  of_isColimit h


/-- Variant of `of_isColimit` for an arbitrary cocone on a diagram `WalkingSpan ⥤ C`. -/
lemma of_isColimit_cocone {D : WalkingSpan ⥤ C} {c : Cocone D} (hc : IsColimit c) :
    IsPushout (D.map WalkingSpan.Hom.fst) (D.map WalkingSpan.Hom.snd)
      (c.ι.app .left) (c.ι.app .right) where
          /-
            C : Type u₁
            inst✝ : CategoryTheory.Category.{v₁, u₁} C
            D : CategoryTheory.Functor CategoryTheory.Limits.WalkingSpan C
            c : CategoryTheory.Limits.Cocone D
            hc : CategoryTheory.Limits.IsColimit c
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.map CategoryTheory.Limits.WalkingS …
          -/
  w := by simp_rw [Cocone.w]
          /-
            🎉 no goals
          -/
  isColimit' := ⟨IsColimit.equivOfNatIsoOfIso _ _ _ (PushoutCocone.isoMk c) hc⟩


lemma hasPushout (h : IsPushout f g inl inr) : HasPushout f g where
  exists_colimit := ⟨⟨h.cocone, h.isColimit⟩⟩


/-- The pushout provided by `HasPushout f g` fits into an `IsPushout`. -/
theorem of_hasPushout (f : Z ⟶ X) (g : Z ⟶ Y) [HasPushout f g] :
    IsPushout f g (pushout.inl f g) (pushout.inr f g) :=
  of_isColimit (colimit.isColimit (span f g))


/-- If `c` is a colimiting binary coproduct cocone, and we have an initial object,
then we have `IsPushout 0 0 c.inl c.inr`
(where each `0` is the unique morphism from the initial object). -/
theorem of_is_coproduct {c : BinaryCofan X Y} (h : Limits.IsColimit c) (t : IsInitial Z) :
    IsPushout (t.to _) (t.to _) c.inl c.inr :=
  of_isColimit
    (isPushoutOfIsInitialIsCoproduct _ _ _ _ t
      (IsColimit.ofIsoColimit h
        (Limits.Cocones.ext (Iso.refl c.pt)
          (by
            /-
              C : Type u₁
              inst✝ : CategoryTheory.Category.{v₁, u₁} C
              Z X Y : C
              c : CategoryTheory.Limits.BinaryCofan X Y
              h : CategoryTheory.Limits.IsColimit c
              t : CategoryTheory.Limits.IsInitial Z
              ⊢ ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Categ …
            -/
            rintro ⟨⟨⟩⟩ <;>
                /-
                  case mk.left
                  C : Type u₁
                  inst✝ : CategoryTheory.Category.{v₁, u₁} C
                  Z X Y : C
                  c : CategoryTheory.Limits.BinaryCofan X Y
                  h : CategoryTheory.Limits.IsColimit c
                  t : CategoryTheory.Limits.IsInitial Z
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.ι.app { as := CategoryTheory.Limit …
                -/
                /-
                  case mk.left
                  C : Type u₁
                  inst✝ : CategoryTheory.Category.{v₁, u₁} C
                  Z X Y : C
                  c : CategoryTheory.Limits.BinaryCofan X Y
                  h : CategoryTheory.Limits.IsColimit c
                  t : CategoryTheory.Limits.IsInitial Z
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp c.inl (CategoryTheory.CategoryStruct. …
                -/
                /-
                  🎉 no goals
                -/
                /-
                  case mk.right
                  C : Type u₁
                  inst✝ : CategoryTheory.Category.{v₁, u₁} C
                  Z X Y : C
                  c : CategoryTheory.Limits.BinaryCofan X Y
                  h : CategoryTheory.Limits.IsColimit c
                  t : CategoryTheory.Limits.IsInitial Z
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp c.inr (CategoryTheory.CategoryStruct. …
                -/
                simp))))
                /-
                  🎉 no goals
                -/


/-- A variant of `of_is_coproduct` that is more useful with `apply`. -/
theorem of_is_coproduct' (h : Limits.IsColimit (BinaryCofan.mk inl inr)) (t : IsInitial Z) :
    IsPushout (t.to _) (t.to _) inl inr :=
  of_is_coproduct h t


theorem of_hasBinaryCoproduct' [HasBinaryCoproduct X Y] [HasInitial C] :
    IsPushout (initial.to _) (initial.to _) (coprod.inl : X ⟶ _) (coprod.inr : Y ⟶ _) :=
  of_is_coproduct (colimit.isColimit _) initialIsInitial


theorem of_hasBinaryCoproduct [HasBinaryCoproduct X Y] [HasZeroObject C] [HasZeroMorphisms C] :
    IsPushout (0 : 0 ⟶ X) (0 : 0 ⟶ Y) coprod.inl coprod.inr := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    inst✝² : CategoryTheory.Limits.HasBinaryCoproduct X Y
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    ⊢ CategoryTheory.IsPushout 0 0 CategoryTheory.Limits.coprod.inl CategoryTheory …
  -/
  convert @of_is_coproduct _ _ 0 X Y _ (colimit.isColimit _) HasZeroObject.zeroIsInitial
        /-
          case h.e'_7.h
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          X Y : C
          inst✝² : CategoryTheory.Limits.HasBinaryCoproduct X Y
          inst✝¹ : CategoryTheory.Limits.HasZeroObject C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          e_4✝ : Eq X ((CategoryTheory.Limits.pair X Y).obj { as := CategoryTheory.Limit …
          ⊢ Eq 0 (CategoryTheory.Limits.HasZeroObject.zeroIsInitial.to ((CategoryTheory. …
        -/
        /-
          🎉 no goals
        -/
    <;> subsingleton
        /-
          🎉 no goals
        -/


/-- Any object at the bottom right of a pushout square is isomorphic to the object at the bottom
right of any other pushout square with the same span. -/
noncomputable def isoIsPushout (h : IsPushout f g inl inr) (h' : IsPushout f g inl' inr') :
    P ≅ P' :=
  IsColimit.coconePointUniqueUpToIso h.isColimit h'.isColimit


@[reassoc (attr := simp)]
theorem inl_isoIsPushout_hom (h : IsPushout f g inl inr) (h' : IsPushout f g inl' inr') :
    inl ≫ (h.isoIsPushout _ _ h').hom = inl' :=
  IsColimit.comp_coconePointUniqueUpToIso_hom h.isColimit h'.isColimit WalkingSpan.left


@[reassoc (attr := simp)]
theorem inr_isoIsPushout_hom (h : IsPushout f g inl inr) (h' : IsPushout f g inl' inr') :
    inr ≫ (h.isoIsPushout _ _ h').hom = inr' :=
  IsColimit.comp_coconePointUniqueUpToIso_hom h.isColimit h'.isColimit WalkingSpan.right


@[reassoc (attr := simp)]
theorem inl_isoIsPushout_inv (h : IsPushout f g inl inr) (h' : IsPushout f g inl' inr') :
    inl' ≫ (h.isoIsPushout _ _ h').inv = inl := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    Z X Y P : C
    f : Quiver.Hom Z X
    g : Quiver.Hom Z Y
    inl : Quiver.Hom X P
    inr : Quiver.Hom Y P
    P' : C
    inl' : Quiver.Hom X P'
    inr' : Quiver.Hom Y P'
    h : CategoryTheory.IsPushout f g inl inr
    h' : CategoryTheory.IsPushout f g inl' inr'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp inl' (CategoryTheory.IsPushout.isoIsP …
  -/
  simp only [Iso.comp_inv_eq, inl_isoIsPushout_hom]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem inr_isoIsPushout_inv (h : IsPushout f g inl inr) (h' : IsPushout f g inl' inr') :
    inr' ≫ (h.isoIsPushout _ _ h').inv = inr := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    Z X Y P : C
    f : Quiver.Hom Z X
    g : Quiver.Hom Z Y
    inl : Quiver.Hom X P
    inr : Quiver.Hom Y P
    P' : C
    inl' : Quiver.Hom X P'
    inr' : Quiver.Hom Y P'
    h : CategoryTheory.IsPushout f g inl inr
    h' : CategoryTheory.IsPushout f g inl' inr'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp inr' (CategoryTheory.IsPushout.isoIsP …
  -/
  simp only [Iso.comp_inv_eq, inr_isoIsPushout_hom]
  /-
    🎉 no goals
  -/


/-- Any object at the top left of a pullback square is
isomorphic to the pullback provided by the `HasLimit` API. -/
noncomputable def isoPushout (h : IsPushout f g inl inr) [HasPushout f g] : P ≅ pushout f g :=
  (colimit.isoColimitCocone ⟨_, h.isColimit⟩).symm


@[reassoc (attr := simp)]
theorem inl_isoPushout_inv (h : IsPushout f g inl inr) [HasPushout f g] :
    pushout.inl _ _ ≫ h.isoPushout.inv = inl := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    Z X Y P : C
    f : Quiver.Hom Z X
    g : Quiver.Hom Z Y
    inl : Quiver.Hom X P
    inr : Quiver.Hom Y P
    h : CategoryTheory.IsPushout f g inl inr
    inst✝ : CategoryTheory.Limits.HasPushout f g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inl f  …
  -/
  dsimp [isoPushout, cocone, CommSq.cocone]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    Z X Y P : C
    f : Quiver.Hom Z X
    g : Quiver.Hom Z Y
    inl : Quiver.Hom X P
    inr : Quiver.Hom Y P
    h : CategoryTheory.IsPushout f g inl inr
    inst✝ : CategoryTheory.Limits.HasPushout f g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inl f  …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem inr_isoPushout_inv (h : IsPushout f g inl inr) [HasPushout f g] :
    pushout.inr _ _ ≫ h.isoPushout.inv = inr := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    Z X Y P : C
    f : Quiver.Hom Z X
    g : Quiver.Hom Z Y
    inl : Quiver.Hom X P
    inr : Quiver.Hom Y P
    h : CategoryTheory.IsPushout f g inl inr
    inst✝ : CategoryTheory.Limits.HasPushout f g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inr f  …
  -/
  dsimp [isoPushout, cocone, CommSq.cocone]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    Z X Y P : C
    f : Quiver.Hom Z X
    g : Quiver.Hom Z Y
    inl : Quiver.Hom X P
    inr : Quiver.Hom Y P
    h : CategoryTheory.IsPushout f g inl inr
    inst✝ : CategoryTheory.Limits.HasPushout f g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inr f  …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem inl_isoPushout_hom (h : IsPushout f g inl inr) [HasPushout f g] :
                                                   /-
                                                     C : Type u₁
                                                     inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                     Z X Y P : C
                                                     f : Quiver.Hom Z X
                                                     g : Quiver.Hom Z Y
                                                     inl : Quiver.Hom X P
                                                     inr : Quiver.Hom Y P
                                                     h : CategoryTheory.IsPushout f g inl inr
                                                     inst✝ : CategoryTheory.Limits.HasPushout f g
                                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp inl h.isoPushout.hom) (CategoryTheory …
                                                   -/
    inl ≫ h.isoPushout.hom = pushout.inl _ _ := by simp [← Iso.eq_comp_inv]
                                                   /-
                                                     🎉 no goals
                                                   -/


@[reassoc (attr := simp)]
theorem inr_isoPushout_hom (h : IsPushout f g inl inr) [HasPushout f g] :
                                                   /-
                                                     C : Type u₁
                                                     inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                     Z X Y P : C
                                                     f : Quiver.Hom Z X
                                                     g : Quiver.Hom Z Y
                                                     inl : Quiver.Hom X P
                                                     inr : Quiver.Hom Y P
                                                     h : CategoryTheory.IsPushout f g inl inr
                                                     inst✝ : CategoryTheory.Limits.HasPushout f g
                                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp inr h.isoPushout.hom) (CategoryTheory …
                                                   -/
    inr ≫ h.isoPushout.hom = pushout.inr _ _ := by simp [← Iso.eq_comp_inv]
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem of_iso_pushout (h : CommSq f g inl inr) [HasPushout f g] (i : P ≅ pushout f g)
    (w₁ : inl ≫ i.hom = pushout.inl _ _) (w₂ : inr ≫ i.hom = pushout.inr _ _) :
      IsPushout f g inl inr :=
  of_isColimit' h
    (Limits.IsColimit.ofIsoColimit (colimit.isColimit _)
      (PushoutCocone.ext (s := PushoutCocone.mk ..) i w₁ w₂).symm)


lemma of_iso (h : IsPushout f g inl inr)
    {Z' X' Y' P' : C} {f' : Z' ⟶ X'} {g' : Z' ⟶ Y'} {inl' : X' ⟶ P'} {inr' : Y' ⟶ P'}
    (e₁ : Z ≅ Z') (e₂ : X ≅ X') (e₃ : Y ≅ Y') (e₄ : P ≅ P')
    (commf : f ≫ e₂.hom = e₁.hom ≫ f')
    (commg : g ≫ e₃.hom = e₁.hom ≫ g')
    (comminl : inl ≫ e₄.hom = e₂.hom ≫ inl')
    (comminr : inr ≫ e₄.hom = e₃.hom ≫ inr') :
    IsPushout f' g' inl' inr' where
  w := by
    rw [← cancel_epi e₁.hom, ← reassoc_of% commf, ← comminl,
      ← reassoc_of% commg, ← comminr, h.w_assoc]
  isColimit' :=
    ⟨(IsColimit.precomposeHomEquiv
        (spanExt e₁ e₂ e₃ commf.symm commg.symm) _).1
          (IsColimit.ofIsoColimit h.isColimit
            (PushoutCocone.ext e₄ comminl comminr))⟩


lemma isIso_inl_iso_of_epi (h : IsPushout f f inl inr) : IsIso inl :=
  h.cocone.isIso_inl_of_epi_of_isColimit h.isColimit


lemma isIso_inr_iso_of_epi (h : IsPushout f f inl inr) : IsIso inr :=
  h.cocone.isIso_inr_of_epi_of_isColimit h.isColimit


theorem flip (h : IsPullback fst snd f g) : IsPullback snd fst g f :=
  of_isLimit (PullbackCone.flipIsLimit h.isLimit)


theorem flip_iff : IsPullback fst snd f g ↔ IsPullback snd fst g f :=
  ⟨flip, flip⟩


/-- The square with `0 : 0 ⟶ 0` on the left and `𝟙 X` on the right is a pullback square. -/
@[simp]
theorem zero_left (X : C) : IsPullback (0 : 0 ⟶ X) (0 : (0 : C) ⟶ 0) (𝟙 X) (0 : 0 ⟶ X) :=
            /-
              C : Type u₁
              inst✝² : CategoryTheory.Category.{v₁, u₁} C
              inst✝¹ : CategoryTheory.Limits.HasZeroObject C
              inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
              X : C
              ⊢ Eq (CategoryTheory.CategoryStruct.comp 0 (CategoryTheory.CategoryStruct.id X …
            -/
  { w := by simp
            /-
              🎉 no goals
            -/
    isLimit' :=
      ⟨{  lift := fun _ => 0
          fac := fun s => by
            simpa [eq_iff_true_of_subsingleton] using
              @PullbackCone.equalizer_ext _ _ _ _ _ _ _ s _ 0 (𝟙 _)
                (by simpa using (PullbackCone.condition s).symm) }⟩ }


/-- The square with `0 : 0 ⟶ 0` on the top and `𝟙 X` on the bottom is a pullback square. -/
@[simp]
theorem zero_top (X : C) : IsPullback (0 : (0 : C) ⟶ 0) (0 : 0 ⟶ X) (0 : 0 ⟶ X) (𝟙 X) :=
  (zero_left X).flip


/-- The square with `0 : 0 ⟶ 0` on the right and `𝟙 X` on the left is a pullback square. -/
@[simp]
theorem zero_right (X : C) : IsPullback (0 : X ⟶ 0) (𝟙 X) (0 : (0 : C) ⟶ 0) (0 : X ⟶ 0) :=
                      /-
                        C : Type u₁
                        inst✝² : CategoryTheory.Category.{v₁, u₁} C
                        inst✝¹ : CategoryTheory.Limits.HasZeroObject C
                        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                        X : C
                        ⊢ CategoryTheory.CommSq 0 (CategoryTheory.CategoryStruct.id X) 0 0
                      -/
  of_iso_pullback (by simp) ((zeroProdIso X).symm ≪≫ (pullbackZeroZeroIso _ _).symm)
                      /-
                        🎉 no goals
                      -/
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Limits.HasZeroObject C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          X : C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.zeroProdIso X …
        -/
        /-
          🎉 no goals
        -/
    (by simp [eq_iff_true_of_subsingleton]) (by simp)
                                                /-
                                                  🎉 no goals
                                                -/


/-- The square with `0 : 0 ⟶ 0` on the bottom and `𝟙 X` on the top is a pullback square. -/
@[simp]
theorem zero_bot (X : C) : IsPullback (𝟙 X) (0 : X ⟶ 0) (0 : X ⟶ 0) (0 : (0 : C) ⟶ 0) :=
  (zero_right X).flip


/-- Paste two pullback squares "vertically" to obtain another pullback square.

The objects in the statement fit into the following diagram:
```
X₁₁ - h₁₁ -> X₁₂
|            |
v₁₁          v₁₂
↓            ↓
X₂₁ - h₂₁ -> X₂₂
|            |
v₂₁          v₂₂
↓            ↓
X₃₁ - h₃₁ -> X₃₂
```
-/
theorem paste_vert {X₁₁ X₁₂ X₂₁ X₂₂ X₃₁ X₃₂ : C} {h₁₁ : X₁₁ ⟶ X₁₂} {h₂₁ : X₂₁ ⟶ X₂₂}
    {h₃₁ : X₃₁ ⟶ X₃₂} {v₁₁ : X₁₁ ⟶ X₂₁} {v₁₂ : X₁₂ ⟶ X₂₂} {v₂₁ : X₂₁ ⟶ X₃₁} {v₂₂ : X₂₂ ⟶ X₃₂}
    (s : IsPullback h₁₁ v₁₁ v₁₂ h₂₁) (t : IsPullback h₂₁ v₂₁ v₂₂ h₃₁) :
    IsPullback h₁₁ (v₁₁ ≫ v₂₁) (v₁₂ ≫ v₂₂) h₃₁ :=
  of_isLimit (pasteHorizIsPullback rfl t.isLimit s.isLimit)


/-- Paste two pullback squares "horizontally" to obtain another pullback square.

The objects in the statement fit into the following diagram:
```
X₁₁ - h₁₁ -> X₁₂ - h₁₂ -> X₁₃
|            |            |
v₁₁          v₁₂          v₁₃
↓            ↓            ↓
X₂₁ - h₂₁ -> X₂₂ - h₂₂ -> X₂₃
```
-/
theorem paste_horiz {X₁₁ X₁₂ X₁₃ X₂₁ X₂₂ X₂₃ : C} {h₁₁ : X₁₁ ⟶ X₁₂} {h₁₂ : X₁₂ ⟶ X₁₃}
    {h₂₁ : X₂₁ ⟶ X₂₂} {h₂₂ : X₂₂ ⟶ X₂₃} {v₁₁ : X₁₁ ⟶ X₂₁} {v₁₂ : X₁₂ ⟶ X₂₂} {v₁₃ : X₁₃ ⟶ X₂₃}
    (s : IsPullback h₁₁ v₁₁ v₁₂ h₂₁) (t : IsPullback h₁₂ v₁₂ v₁₃ h₂₂) :
    IsPullback (h₁₁ ≫ h₁₂) v₁₁ v₁₃ (h₂₁ ≫ h₂₂) :=
  (paste_vert s.flip t.flip).flip


/-- Given a pullback square assembled from a commuting square on the top and
a pullback square on the bottom, the top square is a pullback square.

The objects in the statement fit into the following diagram:
```
X₁₁ - h₁₁ -> X₁₂
|            |
v₁₁          v₁₂
↓            ↓
X₂₁ - h₂₁ -> X₂₂
|            |
v₂₁          v₂₂
↓            ↓
X₃₁ - h₃₁ -> X₃₂
```
-/
theorem of_bot {X₁₁ X₁₂ X₂₁ X₂₂ X₃₁ X₃₂ : C} {h₁₁ : X₁₁ ⟶ X₁₂} {h₂₁ : X₂₁ ⟶ X₂₂} {h₃₁ : X₃₁ ⟶ X₃₂}
    {v₁₁ : X₁₁ ⟶ X₂₁} {v₁₂ : X₁₂ ⟶ X₂₂} {v₂₁ : X₂₁ ⟶ X₃₁} {v₂₂ : X₂₂ ⟶ X₃₂}
    (s : IsPullback h₁₁ (v₁₁ ≫ v₂₁) (v₁₂ ≫ v₂₂) h₃₁) (p : h₁₁ ≫ v₁₂ = v₁₁ ≫ h₂₁)
    (t : IsPullback h₂₁ v₂₁ v₂₂ h₃₁) : IsPullback h₁₁ v₁₁ v₁₂ h₂₁ :=
  of_isLimit (leftSquareIsPullback (PullbackCone.mk h₁₁ _ p) rfl t.isLimit s.isLimit)


/-- Given a pullback square assembled from a commuting square on the left and
a pullback square on the right, the left square is a pullback square.

The objects in the statement fit into the following diagram:
```
X₁₁ - h₁₁ -> X₁₂ - h₁₂ -> X₁₃
|            |            |
v₁₁          v₁₂          v₁₃
↓            ↓            ↓
X₂₁ - h₂₁ -> X₂₂ - h₂₂ -> X₂₃
```
-/
theorem of_right {X₁₁ X₁₂ X₁₃ X₂₁ X₂₂ X₂₃ : C} {h₁₁ : X₁₁ ⟶ X₁₂} {h₁₂ : X₁₂ ⟶ X₁₃} {h₂₁ : X₂₁ ⟶ X₂₂}
    {h₂₂ : X₂₂ ⟶ X₂₃} {v₁₁ : X₁₁ ⟶ X₂₁} {v₁₂ : X₁₂ ⟶ X₂₂} {v₁₃ : X₁₃ ⟶ X₂₃}
    (s : IsPullback (h₁₁ ≫ h₁₂) v₁₁ v₁₃ (h₂₁ ≫ h₂₂)) (p : h₁₁ ≫ v₁₂ = v₁₁ ≫ h₂₁)
    (t : IsPullback h₁₂ v₁₂ v₁₃ h₂₂) : IsPullback h₁₁ v₁₁ v₁₂ h₂₁ :=
  (of_bot s.flip p.symm t.flip).flip


theorem paste_vert_iff {X₁₁ X₁₂ X₂₁ X₂₂ X₃₁ X₃₂ : C} {h₁₁ : X₁₁ ⟶ X₁₂} {h₂₁ : X₂₁ ⟶ X₂₂}
    {h₃₁ : X₃₁ ⟶ X₃₂} {v₁₁ : X₁₁ ⟶ X₂₁} {v₁₂ : X₁₂ ⟶ X₂₂} {v₂₁ : X₂₁ ⟶ X₃₁} {v₂₂ : X₂₂ ⟶ X₃₂}
    (s : IsPullback h₂₁ v₂₁ v₂₂ h₃₁) (e : h₁₁ ≫ v₁₂ = v₁₁ ≫ h₂₁) :
    IsPullback h₁₁ (v₁₁ ≫ v₂₁) (v₁₂ ≫ v₂₂) h₃₁ ↔ IsPullback h₁₁ v₁₁ v₁₂ h₂₁ :=
  ⟨fun h => h.of_bot e s, fun h => h.paste_vert s⟩


theorem paste_horiz_iff {X₁₁ X₁₂ X₁₃ X₂₁ X₂₂ X₂₃ : C} {h₁₁ : X₁₁ ⟶ X₁₂} {h₁₂ : X₁₂ ⟶ X₁₃}
    {h₂₁ : X₂₁ ⟶ X₂₂} {h₂₂ : X₂₂ ⟶ X₂₃} {v₁₁ : X₁₁ ⟶ X₂₁} {v₁₂ : X₁₂ ⟶ X₂₂} {v₁₃ : X₁₃ ⟶ X₂₃}
    (s : IsPullback h₁₂ v₁₂ v₁₃ h₂₂) (e : h₁₁ ≫ v₁₂ = v₁₁ ≫ h₂₁) :
    IsPullback (h₁₁ ≫ h₁₂) v₁₁ v₁₃ (h₂₁ ≫ h₂₂) ↔ IsPullback h₁₁ v₁₁ v₁₂ h₂₁ :=
  ⟨fun h => h.of_right e s, fun h => h.paste_horiz s⟩


/-- Variant of `IsPullback.of_right` where `h₁₁` is induced from a morphism `h₁₃ : X₁₁ ⟶ X₁₃`, and
the universal property of the right square.

The objects fit in the following diagram:
```
X₁₁ - h₁₁ -> X₁₂ - h₁₂ -> X₁₃
|            |            |
v₁₁          v₁₂          v₁₃
↓            ↓            ↓
X₂₁ - h₂₁ -> X₂₂ - h₂₂ -> X₂₃
```
-/
theorem of_right' {X₁₁ X₁₂ X₁₃ X₂₁ X₂₂ X₂₃ : C} {h₁₂ : X₁₂ ⟶ X₁₃} {h₂₁ : X₂₁ ⟶ X₂₂}
    {h₂₂ : X₂₂ ⟶ X₂₃} {h₁₃ : X₁₁ ⟶ X₁₃} {v₁₁ : X₁₁ ⟶ X₂₁} {v₁₂ : X₁₂ ⟶ X₂₂} {v₁₃ : X₁₃ ⟶ X₂₃}
    (s : IsPullback h₁₃ v₁₁ v₁₃ (h₂₁ ≫ h₂₂)) (t : IsPullback h₁₂ v₁₂ v₁₃ h₂₂) :
                                           /-
                                             C : Type u₁
                                             inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                             P X Y Z : C
                                             fst : Quiver.Hom P X
                                             snd : Quiver.Hom P Y
                                             f : Quiver.Hom X Z
                                             g : Quiver.Hom Y Z
                                             X₁₁ X₁₂ X₁₃ X₂₁ X₂₂ X₂₃ : C
                                             h₁₂ : Quiver.Hom X₁₂ X₁₃
                                             h₂₁ : Quiver.Hom X₂₁ X₂₂
                                             h₂₂ : Quiver.Hom X₂₂ X₂₃
                                             h₁₃ : Quiver.Hom X₁₁ X₁₃
                                             v₁₁ : Quiver.Hom X₁₁ X₂₁
                                             v₁₂ : Quiver.Hom X₁₂ X₂₂
                                             v₁₃ : Quiver.Hom X₁₃ X₂₃
                                             s : CategoryTheory.IsPullback h₁₃ v₁₁ v₁₃ (CategoryTheory.CategoryStruct.comp  …
                                             t : CategoryTheory.IsPullback h₁₂ v₁₂ v₁₃ h₂₂
                                             ⊢ Eq (CategoryTheory.CategoryStruct.comp h₁₃ v₁₃) (CategoryTheory.CategoryStru …
                                           -/
    IsPullback (t.lift h₁₃ (v₁₁ ≫ h₂₁) (by rw [s.w, Category.assoc])) v₁₁ v₁₂ h₂₁ :=
                                           /-
                                             🎉 no goals
                                           -/
  of_right ((t.lift_fst _ _ _) ▸ s) (t.lift_snd _ _ _) t


/-- Variant of `IsPullback.of_bot`, where `v₁₁` is induced from a morphism `v₃₁ : X₁₁ ⟶ X₃₁`, and
the universal property of the bottom square.

The objects in the statement fit into the following diagram:
```
X₁₁ - h₁₁ -> X₁₂
|            |
v₁₁          v₁₂
↓            ↓
X₂₁ - h₂₁ -> X₂₂
|            |
v₂₁          v₂₂
↓            ↓
X₃₁ - h₃₁ -> X₃₂
```
-/
theorem of_bot' {X₁₁ X₁₂ X₂₁ X₂₂ X₃₁ X₃₂ : C} {h₁₁ : X₁₁ ⟶ X₁₂} {h₂₁ : X₂₁ ⟶ X₂₂}
    {h₃₁ : X₃₁ ⟶ X₃₂} {v₃₁ : X₁₁ ⟶ X₃₁} {v₁₂ : X₁₂ ⟶ X₂₂} {v₂₁ : X₂₁ ⟶ X₃₁} {v₂₂ : X₂₂ ⟶ X₃₂}
    (s : IsPullback h₁₁ v₃₁ (v₁₂ ≫ v₂₂) h₃₁) (t : IsPullback h₂₁ v₂₁ v₂₂ h₃₁) :
                                               /-
                                                 C : Type u₁
                                                 inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                                 P X Y Z : C
                                                 fst : Quiver.Hom P X
                                                 snd : Quiver.Hom P Y
                                                 f : Quiver.Hom X Z
                                                 g : Quiver.Hom Y Z
                                                 X₁₁ X₁₂ X₂₁ X₂₂ X₃₁ X₃₂ : C
                                                 h₁₁ : Quiver.Hom X₁₁ X₁₂
                                                 h₂₁ : Quiver.Hom X₂₁ X₂₂
                                                 h₃₁ : Quiver.Hom X₃₁ X₃₂
                                                 v₃₁ : Quiver.Hom X₁₁ X₃₁
                                                 v₁₂ : Quiver.Hom X₁₂ X₂₂
                                                 v₂₁ : Quiver.Hom X₂₁ X₃₁
                                                 v₂₂ : Quiver.Hom X₂₂ X₃₂
                                                 s : CategoryTheory.IsPullback h₁₁ v₃₁ (CategoryTheory.CategoryStruct.comp v₁₂  …
                                                 t : CategoryTheory.IsPullback h₂₁ v₂₁ v₂₂ h₃₁
                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp h …
                                               -/
    IsPullback h₁₁ (t.lift (h₁₁ ≫ v₁₂) v₃₁ (by rw [Category.assoc, s.w])) v₁₂ h₂₁ :=
                                               /-
                                                 🎉 no goals
                                               -/
                                      /-
                                        C : Type u₁
                                        inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                        X₁₁ X₁₂ X₂₁ X₂₂ X₃₁ X₃₂ : C
                                        h₁₁ : Quiver.Hom X₁₁ X₁₂
                                        h₂₁ : Quiver.Hom X₂₁ X₂₂
                                        h₃₁ : Quiver.Hom X₃₁ X₃₂
                                        v₃₁ : Quiver.Hom X₁₁ X₃₁
                                        v₁₂ : Quiver.Hom X₁₂ X₂₂
                                        v₂₁ : Quiver.Hom X₂₁ X₃₁
                                        v₂₂ : Quiver.Hom X₂₂ X₃₂
                                        s : CategoryTheory.IsPullback h₁₁ v₃₁ (CategoryTheory.CategoryStruct.comp v₁₂  …
                                        t : CategoryTheory.IsPullback h₂₁ v₂₁ v₂₂ h₃₁
                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp h₁₁ v₁₂) (CategoryTheory.CategoryStru …
                                      -/
  of_bot ((t.lift_snd _ _ _) ▸ s) (by simp only [lift_fst]) t
                                      /-
                                        🎉 no goals
                                      -/


theorem of_isBilimit {b : BinaryBicone X Y} (h : b.IsBilimit) :
    IsPullback b.fst b.snd (0 : X ⟶ 0) (0 : Y ⟶ 0) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    b : CategoryTheory.Limits.BinaryBicone X Y
    h : b.IsBilimit
    ⊢ CategoryTheory.IsPullback b.fst b.snd 0 0
  -/
  convert IsPullback.of_is_product' h.isLimit HasZeroObject.zeroIsTerminal
        /-
          case h.e'_9
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          X Y : C
          inst✝¹ : CategoryTheory.Limits.HasZeroObject C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          b : CategoryTheory.Limits.BinaryBicone X Y
          h : b.IsBilimit
          ⊢ Eq 0 (CategoryTheory.Limits.HasZeroObject.zeroIsTerminal.from X)
        -/
        /-
          🎉 no goals
        -/
    <;> subsingleton
        /-
          🎉 no goals
        -/


@[simp]
theorem of_has_biproduct (X Y : C) [HasBinaryBiproduct X Y] :
    IsPullback biprod.fst biprod.snd (0 : X ⟶ 0) (0 : Y ⟶ 0) :=
  of_isBilimit (BinaryBiproduct.isBilimit X Y)


theorem inl_snd' {b : BinaryBicone X Y} (h : b.IsBilimit) :
    IsPullback b.inl (0 : X ⟶ 0) b.snd (0 : 0 ⟶ Y) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    b : CategoryTheory.Limits.BinaryBicone X Y
    h : b.IsBilimit
    ⊢ CategoryTheory.IsPullback b.inl 0 b.snd 0
  -/
  refine of_right ?_ (by simp) (of_isBilimit h)
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    b : CategoryTheory.Limits.BinaryBicone X Y
    h : b.IsBilimit
    ⊢ CategoryTheory.IsPullback (CategoryTheory.CategoryStruct.comp b.inl b.fst) 0 …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The square
```
  X --inl--> X ⊞ Y
  |            |
  0           snd
  |            |
  v            v
  0 ---0-----> Y
```
is a pullback square.
-/
@[simp]
theorem inl_snd (X Y : C) [HasBinaryBiproduct X Y] :
    IsPullback biprod.inl (0 : X ⟶ 0) biprod.snd (0 : 0 ⟶ Y) :=
  inl_snd' (BinaryBiproduct.isBilimit X Y)


theorem inr_fst' {b : BinaryBicone X Y} (h : b.IsBilimit) :
    IsPullback b.inr (0 : Y ⟶ 0) b.fst (0 : 0 ⟶ X) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    b : CategoryTheory.Limits.BinaryBicone X Y
    h : b.IsBilimit
    ⊢ CategoryTheory.IsPullback b.inr 0 b.fst 0
  -/
  apply flip
  /-
    case h
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    b : CategoryTheory.Limits.BinaryBicone X Y
    h : b.IsBilimit
    ⊢ CategoryTheory.IsPullback 0 b.inr 0 b.fst
  -/
  refine of_bot ?_ (by simp) (of_isBilimit h)
  /-
    case h
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    b : CategoryTheory.Limits.BinaryBicone X Y
    h : b.IsBilimit
    ⊢ CategoryTheory.IsPullback 0 (CategoryTheory.CategoryStruct.comp b.inr b.snd) …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The square
```
  Y --inr--> X ⊞ Y
  |            |
  0           fst
  |            |
  v            v
  0 ---0-----> X
```
is a pullback square.
-/
@[simp]
theorem inr_fst (X Y : C) [HasBinaryBiproduct X Y] :
    IsPullback biprod.inr (0 : Y ⟶ 0) biprod.fst (0 : 0 ⟶ X) :=
  inr_fst' (BinaryBiproduct.isBilimit X Y)


theorem of_is_bilimit' {b : BinaryBicone X Y} (h : b.IsBilimit) :
    IsPullback (0 : 0 ⟶ X) (0 : 0 ⟶ Y) b.inl b.inr := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    b : CategoryTheory.Limits.BinaryBicone X Y
    h : b.IsBilimit
    ⊢ CategoryTheory.IsPullback 0 0 b.inl b.inr
  -/
  refine IsPullback.of_right ?_ (by simp) (IsPullback.inl_snd' h).flip
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    b : CategoryTheory.Limits.BinaryBicone X Y
    h : b.IsBilimit
    ⊢ CategoryTheory.IsPullback (CategoryTheory.CategoryStruct.comp 0 0) 0 0 (Cate …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem of_hasBinaryBiproduct (X Y : C) [HasBinaryBiproduct X Y] :
    IsPullback (0 : 0 ⟶ X) (0 : 0 ⟶ Y) biprod.inl biprod.inr :=
  of_is_bilimit' (BinaryBiproduct.isBilimit X Y)


instance hasPullback_biprod_fst_biprod_snd [HasBinaryBiproduct X Y] :
    HasPullback (biprod.inl : X ⟶ _) (biprod.inr : Y ⟶ _) :=
  HasLimit.mk ⟨_, (of_hasBinaryBiproduct X Y).isLimit⟩


/-- The pullback of `biprod.inl` and `biprod.inr` is the zero object. -/
def pullbackBiprodInlBiprodInr [HasBinaryBiproduct X Y] :
    pullback (biprod.inl : X ⟶ _) (biprod.inr : Y ⟶ _) ≅ 0 :=
  limit.isoLimitCone ⟨_, (of_hasBinaryBiproduct X Y).isLimit⟩


theorem op (h : IsPullback fst snd f g) : IsPushout g.op f.op snd.op fst.op :=
  IsPushout.of_isColimit
    (IsColimit.ofIsoColimit (Limits.PullbackCone.isLimitEquivIsColimitOp h.flip.cone h.flip.isLimit)
      h.toCommSq.flip.coneOp)


theorem unop {P X Y Z : Cᵒᵖ} {fst : P ⟶ X} {snd : P ⟶ Y} {f : X ⟶ Z} {g : Y ⟶ Z}
    (h : IsPullback fst snd f g) : IsPushout g.unop f.unop snd.unop fst.unop :=
  IsPushout.of_isColimit
    (IsColimit.ofIsoColimit
      (Limits.PullbackCone.isLimitEquivIsColimitUnop h.flip.cone h.flip.isLimit)
      h.toCommSq.flip.coneUnop)


theorem of_vert_isIso [IsIso snd] [IsIso f] (sq : CommSq fst snd f g) : IsPullback fst snd f g :=
  IsPullback.flip (of_horiz_isIso sq.flip)


                                                                              /-
                                                                                C : Type u₁
                                                                                inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                                                                X Z : C
                                                                                f : Quiver.Hom X Z
                                                                                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id X)  …
                                                                              -/
lemma of_id_fst : IsPullback (𝟙 _) f f (𝟙 _) := IsPullback.of_horiz_isIso ⟨by simp⟩
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


                                                                             /-
                                                                               C : Type u₁
                                                                               inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                                                               X Z : C
                                                                               f : Quiver.Hom X Z
                                                                               ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.id Z …
                                                                             -/
lemma of_id_snd : IsPullback f (𝟙 _) (𝟙 _) f := IsPullback.of_vert_isIso ⟨by simp⟩
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


/-- The following diagram is a pullback
```
X --f--> Z
|        |
id       id
v        v
X --f--> Z
```
-/
lemma id_vert (f : X ⟶ Z) : IsPullback f (𝟙 X) (𝟙 Z) f :=
                    /-
                      C : Type u₁
                      inst✝ : CategoryTheory.Category.{v₁, u₁} C
                      X Z : C
                      f : Quiver.Hom X Z
                      ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.id Z …
                    -/
  of_vert_isIso ⟨by simp only [Category.id_comp, Category.comp_id]⟩
                    /-
                      🎉 no goals
                    -/


/-- The following diagram is a pullback
```
X --id--> X
|         |
f         f
v         v
Z --id--> Z
```
-/
lemma id_horiz (f : X ⟶ Z) : IsPullback (𝟙 X) f f (𝟙 Z) :=
                     /-
                       C : Type u₁
                       inst✝ : CategoryTheory.Category.{v₁, u₁} C
                       X Z : C
                       f : Quiver.Hom X Z
                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id X)  …
                     -/
  of_horiz_isIso ⟨by simp only [Category.id_comp, Category.comp_id]⟩
                     /-
                       🎉 no goals
                     -/


theorem flip (h : IsPushout f g inl inr) : IsPushout g f inr inl :=
  of_isColimit (PushoutCocone.flipIsColimit h.isColimit)


theorem flip_iff : IsPushout f g inl inr ↔ IsPushout g f inr inl :=
  ⟨flip, flip⟩


/-- The square with `0 : 0 ⟶ 0` on the right and `𝟙 X` on the left is a pushout square. -/
@[simp]
theorem zero_right (X : C) : IsPushout (0 : X ⟶ 0) (𝟙 X) (0 : (0 : C) ⟶ 0) (0 : X ⟶ 0) :=
            /-
              C : Type u₁
              inst✝² : CategoryTheory.Category.{v₁, u₁} C
              inst✝¹ : CategoryTheory.Limits.HasZeroObject C
              inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
              X : C
              ⊢ Eq (CategoryTheory.CategoryStruct.comp 0 0) (CategoryTheory.CategoryStruct.c …
            -/
  { w := by simp
            /-
              🎉 no goals
            -/
    isColimit' :=
      ⟨{  desc := fun _ => 0
          fac := fun s => by
            have c :=
              @PushoutCocone.coequalizer_ext _ _ _ _ _ _ _ s _ 0 (𝟙 _)
                (by simp [eq_iff_true_of_subsingleton]) (by simpa using PushoutCocone.condition s)
            /-
              C : Type u₁
              inst✝² : CategoryTheory.Category.{v₁, u₁} C
              inst✝¹ : CategoryTheory.Limits.HasZeroObject C
              inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
              X : C
              s : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.span 0 (CategoryTheory …
              c : ∀ (j : CategoryTheory.Limits.WalkingSpan), Eq (CategoryTheory.CategoryStru …
              ⊢ ∀ (j : CategoryTheory.Limits.WalkingSpan), Eq (CategoryTheory.CategoryStruct …
            -/
            dsimp at c
            /-
              C : Type u₁
              inst✝² : CategoryTheory.Category.{v₁, u₁} C
              inst✝¹ : CategoryTheory.Limits.HasZeroObject C
              inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
              X : C
              s : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.span 0 (CategoryTheory …
              c : ∀ (j : CategoryTheory.Limits.WalkingSpan), Eq (CategoryTheory.CategoryStru …
              ⊢ ∀ (j : CategoryTheory.Limits.WalkingSpan), Eq (CategoryTheory.CategoryStruct …
            -/
            simpa using c }⟩ }
            /-
              🎉 no goals
            -/


/-- The square with `0 : 0 ⟶ 0` on the bottom and `𝟙 X` on the top is a pushout square. -/
@[simp]
theorem zero_bot (X : C) : IsPushout (𝟙 X) (0 : X ⟶ 0) (0 : X ⟶ 0) (0 : (0 : C) ⟶ 0) :=
  (zero_right X).flip


/-- The square with `0 : 0 ⟶ 0` on the right left `𝟙 X` on the right is a pushout square. -/
@[simp]
theorem zero_left (X : C) : IsPushout (0 : 0 ⟶ X) (0 : (0 : C) ⟶ 0) (𝟙 X) (0 : 0 ⟶ X) :=
                     /-
                       C : Type u₁
                       inst✝² : CategoryTheory.Category.{v₁, u₁} C
                       inst✝¹ : CategoryTheory.Limits.HasZeroObject C
                       inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                       X : C
                       ⊢ CategoryTheory.CommSq 0 0 (CategoryTheory.CategoryStruct.id X) 0
                     -/
                     /-
                       🎉 no goals
                     -/
  of_iso_pushout (by simp) ((coprodZeroIso X).symm ≪≫ (pushoutZeroZeroIso _ _).symm) (by simp)
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Limits.HasZeroObject C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          X : C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp 0 ((CategoryTheory.Limits.coprodZeroI …
        -/
    (by simp [eq_iff_true_of_subsingleton])
        /-
          🎉 no goals
        -/


/-- The square with `0 : 0 ⟶ 0` on the top and `𝟙 X` on the bottom is a pushout square. -/
@[simp]
theorem zero_top (X : C) : IsPushout (0 : (0 : C) ⟶ 0) (0 : 0 ⟶ X) (0 : 0 ⟶ X) (𝟙 X) :=
  (zero_left X).flip


/-- Paste two pushout squares "vertically" to obtain another pushout square.

The objects in the statement fit into the following diagram:
```
X₁₁ - h₁₁ -> X₁₂
|            |
v₁₁          v₁₂
↓            ↓
X₂₁ - h₂₁ -> X₂₂
|            |
v₂₁          v₂₂
↓            ↓
X₃₁ - h₃₁ -> X₃₂
```
-/
theorem paste_vert {X₁₁ X₁₂ X₂₁ X₂₂ X₃₁ X₃₂ : C} {h₁₁ : X₁₁ ⟶ X₁₂} {h₂₁ : X₂₁ ⟶ X₂₂}
    {h₃₁ : X₃₁ ⟶ X₃₂} {v₁₁ : X₁₁ ⟶ X₂₁} {v₁₂ : X₁₂ ⟶ X₂₂} {v₂₁ : X₂₁ ⟶ X₃₁} {v₂₂ : X₂₂ ⟶ X₃₂}
    (s : IsPushout h₁₁ v₁₁ v₁₂ h₂₁) (t : IsPushout h₂₁ v₂₁ v₂₂ h₃₁) :
    IsPushout h₁₁ (v₁₁ ≫ v₂₁) (v₁₂ ≫ v₂₂) h₃₁ :=
  of_isColimit (pasteHorizIsPushout rfl s.isColimit t.isColimit)


/-- Paste two pushout squares "horizontally" to obtain another pushout square.

The objects in the statement fit into the following diagram:
```
X₁₁ - h₁₁ -> X₁₂ - h₁₂ -> X₁₃
|            |            |
v₁₁          v₁₂          v₁₃
↓            ↓            ↓
X₂₁ - h₂₁ -> X₂₂ - h₂₂ -> X₂₃
```
-/
theorem paste_horiz {X₁₁ X₁₂ X₁₃ X₂₁ X₂₂ X₂₃ : C} {h₁₁ : X₁₁ ⟶ X₁₂} {h₁₂ : X₁₂ ⟶ X₁₃}
    {h₂₁ : X₂₁ ⟶ X₂₂} {h₂₂ : X₂₂ ⟶ X₂₃} {v₁₁ : X₁₁ ⟶ X₂₁} {v₁₂ : X₁₂ ⟶ X₂₂} {v₁₃ : X₁₃ ⟶ X₂₃}
    (s : IsPushout h₁₁ v₁₁ v₁₂ h₂₁) (t : IsPushout h₁₂ v₁₂ v₁₃ h₂₂) :
    IsPushout (h₁₁ ≫ h₁₂) v₁₁ v₁₃ (h₂₁ ≫ h₂₂) :=
  (paste_vert s.flip t.flip).flip


/-- Given a pushout square assembled from a pushout square on the top and
a commuting square on the bottom, the bottom square is a pushout square.

The objects in the statement fit into the following diagram:
```
X₁₁ - h₁₁ -> X₁₂
|            |
v₁₁          v₁₂
↓            ↓
X₂₁ - h₂₁ -> X₂₂
|            |
v₂₁          v₂₂
↓            ↓
X₃₁ - h₃₁ -> X₃₂
```
-/
theorem of_top {X₁₁ X₁₂ X₂₁ X₂₂ X₃₁ X₃₂ : C} {h₁₁ : X₁₁ ⟶ X₁₂} {h₂₁ : X₂₁ ⟶ X₂₂} {h₃₁ : X₃₁ ⟶ X₃₂}
    {v₁₁ : X₁₁ ⟶ X₂₁} {v₁₂ : X₁₂ ⟶ X₂₂} {v₂₁ : X₂₁ ⟶ X₃₁} {v₂₂ : X₂₂ ⟶ X₃₂}
    (s : IsPushout h₁₁ (v₁₁ ≫ v₂₁) (v₁₂ ≫ v₂₂) h₃₁) (p : h₂₁ ≫ v₂₂ = v₂₁ ≫ h₃₁)
    (t : IsPushout h₁₁ v₁₁ v₁₂ h₂₁) : IsPushout h₂₁ v₂₁ v₂₂ h₃₁ :=
  of_isColimit <| rightSquareIsPushout
    (PushoutCocone.mk _ _ p) (cocone_inr _) t.isColimit s.isColimit


/-- Given a pushout square assembled from a pushout square on the left and
a commuting square on the right, the right square is a pushout square.

The objects in the statement fit into the following diagram:
```
X₁₁ - h₁₁ -> X₁₂ - h₁₂ -> X₁₃
|            |            |
v₁₁          v₁₂          v₁₃
↓            ↓            ↓
X₂₁ - h₂₁ -> X₂₂ - h₂₂ -> X₂₃
```
-/
theorem of_left {X₁₁ X₁₂ X₁₃ X₂₁ X₂₂ X₂₃ : C} {h₁₁ : X₁₁ ⟶ X₁₂} {h₁₂ : X₁₂ ⟶ X₁₃} {h₂₁ : X₂₁ ⟶ X₂₂}
    {h₂₂ : X₂₂ ⟶ X₂₃} {v₁₁ : X₁₁ ⟶ X₂₁} {v₁₂ : X₁₂ ⟶ X₂₂} {v₁₃ : X₁₃ ⟶ X₂₃}
    (s : IsPushout (h₁₁ ≫ h₁₂) v₁₁ v₁₃ (h₂₁ ≫ h₂₂)) (p : h₁₂ ≫ v₁₃ = v₁₂ ≫ h₂₂)
    (t : IsPushout h₁₁ v₁₁ v₁₂ h₂₁) : IsPushout h₁₂ v₁₂ v₁₃ h₂₂ :=
  (of_top s.flip p.symm t.flip).flip


theorem paste_vert_iff {X₁₁ X₁₂ X₂₁ X₂₂ X₃₁ X₃₂ : C} {h₁₁ : X₁₁ ⟶ X₁₂} {h₂₁ : X₂₁ ⟶ X₂₂}
    {h₃₁ : X₃₁ ⟶ X₃₂} {v₁₁ : X₁₁ ⟶ X₂₁} {v₁₂ : X₁₂ ⟶ X₂₂} {v₂₁ : X₂₁ ⟶ X₃₁} {v₂₂ : X₂₂ ⟶ X₃₂}
    (s : IsPushout h₁₁ v₁₁ v₁₂ h₂₁) (e : h₂₁ ≫ v₂₂ = v₂₁ ≫ h₃₁) :
    IsPushout h₁₁ (v₁₁ ≫ v₂₁) (v₁₂ ≫ v₂₂) h₃₁ ↔ IsPushout h₂₁ v₂₁ v₂₂ h₃₁ :=
  ⟨fun h => h.of_top e s, s.paste_vert⟩


theorem paste_horiz_iff {X₁₁ X₁₂ X₁₃ X₂₁ X₂₂ X₂₃ : C} {h₁₁ : X₁₁ ⟶ X₁₂} {h₁₂ : X₁₂ ⟶ X₁₃}
    {h₂₁ : X₂₁ ⟶ X₂₂} {h₂₂ : X₂₂ ⟶ X₂₃} {v₁₁ : X₁₁ ⟶ X₂₁} {v₁₂ : X₁₂ ⟶ X₂₂} {v₁₃ : X₁₃ ⟶ X₂₃}
    (s : IsPushout h₁₁ v₁₁ v₁₂ h₂₁) (e : h₁₂ ≫ v₁₃ = v₁₂ ≫ h₂₂) :
    IsPushout (h₁₁ ≫ h₁₂) v₁₁ v₁₃ (h₂₁ ≫ h₂₂) ↔ IsPushout h₁₂ v₁₂ v₁₃ h₂₂ :=
  ⟨fun h => h.of_left e s, s.paste_horiz⟩


/-- Variant of `IsPushout.of_top` where `v₂₂` is induced from a morphism `v₁₃ : X₁₂ ⟶ X₃₂`, and
the universal property of the top square.

The objects in the statement fit into the following diagram:
```
X₁₁ - h₁₁ -> X₁₂
|            |
v₁₁          v₁₂
↓            ↓
X₂₁ - h₂₁ -> X₂₂
|            |
v₂₁          v₂₂
↓            ↓
X₃₁ - h₃₁ -> X₃₂
```
-/
theorem of_top' {X₁₁ X₁₂ X₂₁ X₂₂ X₃₁ X₃₂ : C} {h₁₁ : X₁₁ ⟶ X₁₂} {h₂₁ : X₂₁ ⟶ X₂₂} {h₃₁ : X₃₁ ⟶ X₃₂}
    {v₁₁ : X₁₁ ⟶ X₂₁} {v₁₂ : X₁₂ ⟶ X₂₂} {v₁₃ : X₁₂ ⟶ X₃₂} {v₂₁ : X₂₁ ⟶ X₃₁}
    (s : IsPushout h₁₁ (v₁₁ ≫ v₂₁) v₁₃ h₃₁) (t : IsPushout h₁₁ v₁₁ v₁₂ h₂₁) :
                                                    /-
                                                      C : Type u₁
                                                      inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                                      Z X Y P : C
                                                      f : Quiver.Hom Z X
                                                      g : Quiver.Hom Z Y
                                                      inl : Quiver.Hom X P
                                                      inr : Quiver.Hom Y P
                                                      X₁₁ X₁₂ X₂₁ X₂₂ X₃₁ X₃₂ : C
                                                      h₁₁ : Quiver.Hom X₁₁ X₁₂
                                                      h₂₁ : Quiver.Hom X₂₁ X₂₂
                                                      h₃₁ : Quiver.Hom X₃₁ X₃₂
                                                      v₁₁ : Quiver.Hom X₁₁ X₂₁
                                                      v₁₂ : Quiver.Hom X₁₂ X₂₂
                                                      v₁₃ : Quiver.Hom X₁₂ X₃₂
                                                      v₂₁ : Quiver.Hom X₂₁ X₃₁
                                                      s : CategoryTheory.IsPushout h₁₁ (CategoryTheory.CategoryStruct.comp v₁₁ v₂₁)  …
                                                      t : CategoryTheory.IsPushout h₁₁ v₁₁ v₁₂ h₂₁
                                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp h₁₁ v₁₃) (CategoryTheory.CategoryStru …
                                                    -/
      IsPushout h₂₁ v₂₁ (t.desc v₁₃ (v₂₁ ≫ h₃₁) (by rw [s.w, Category.assoc])) h₃₁ :=
                                                    /-
                                                      🎉 no goals
                                                    -/
  of_top ((t.inl_desc _ _ _).symm ▸ s) (t.inr_desc _ _ _) t


/-- Variant of `IsPushout.of_right` where `h₂₂` is induced from a morphism `h₂₃ : X₂₁ ⟶ X₂₃`, and
the universal property of the left square.

The objects in the statement fit into the following diagram:
```
X₁₁ - h₁₁ -> X₁₂ - h₁₂ -> X₁₃
|            |            |
v₁₁          v₁₂          v₁₃
↓            ↓            ↓
X₂₁ - h₂₁ -> X₂₂ - h₂₂ -> X₂₃
```
-/
theorem of_left' {X₁₁ X₁₂ X₁₃ X₂₁ X₂₂ X₂₃ : C} {h₁₁ : X₁₁ ⟶ X₁₂} {h₁₂ : X₁₂ ⟶ X₁₃} {h₂₁ : X₂₁ ⟶ X₂₂}
    {h₂₃ : X₂₁ ⟶ X₂₃} {v₁₁ : X₁₁ ⟶ X₂₁} {v₁₂ : X₁₂ ⟶ X₂₂} {v₁₃ : X₁₃ ⟶ X₂₃}
    (s : IsPushout (h₁₁ ≫ h₁₂) v₁₁ v₁₃ h₂₃) (t : IsPushout h₁₁ v₁₁ v₁₂ h₂₁) :
                                                      /-
                                                        C : Type u₁
                                                        inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                                        Z X Y P : C
                                                        f : Quiver.Hom Z X
                                                        g : Quiver.Hom Z Y
                                                        inl : Quiver.Hom X P
                                                        inr : Quiver.Hom Y P
                                                        X₁₁ X₁₂ X₁₃ X₂₁ X₂₂ X₂₃ : C
                                                        h₁₁ : Quiver.Hom X₁₁ X₁₂
                                                        h₁₂ : Quiver.Hom X₁₂ X₁₃
                                                        h₂₁ : Quiver.Hom X₂₁ X₂₂
                                                        h₂₃ : Quiver.Hom X₂₁ X₂₃
                                                        v₁₁ : Quiver.Hom X₁₁ X₂₁
                                                        v₁₂ : Quiver.Hom X₁₂ X₂₂
                                                        v₁₃ : Quiver.Hom X₁₃ X₂₃
                                                        s : CategoryTheory.IsPushout (CategoryTheory.CategoryStruct.comp h₁₁ h₁₂) v₁₁  …
                                                        t : CategoryTheory.IsPushout h₁₁ v₁₁ v₁₂ h₂₁
                                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp h₁₁ (CategoryTheory.CategoryStruct.co …
                                                      -/
    IsPushout h₁₂ v₁₂ v₁₃ (t.desc (h₁₂ ≫ v₁₃) h₂₃ (by rw [← Category.assoc, s.w])) :=
                                                      /-
                                                        🎉 no goals
                                                      -/
                                            /-
                                              C : Type u₁
                                              inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                              X₁₁ X₁₂ X₁₃ X₂₁ X₂₂ X₂₃ : C
                                              h₁₁ : Quiver.Hom X₁₁ X₁₂
                                              h₁₂ : Quiver.Hom X₁₂ X₁₃
                                              h₂₁ : Quiver.Hom X₂₁ X₂₂
                                              h₂₃ : Quiver.Hom X₂₁ X₂₃
                                              v₁₁ : Quiver.Hom X₁₁ X₂₁
                                              v₁₂ : Quiver.Hom X₁₂ X₂₂
                                              v₁₃ : Quiver.Hom X₁₃ X₂₃
                                              s : CategoryTheory.IsPushout (CategoryTheory.CategoryStruct.comp h₁₁ h₁₂) v₁₁  …
                                              t : CategoryTheory.IsPushout h₁₁ v₁₁ v₁₂ h₂₁
                                              ⊢ Eq (CategoryTheory.CategoryStruct.comp h₁₂ v₁₃) (CategoryTheory.CategoryStru …
                                            -/
  of_left ((t.inr_desc _ _ _).symm ▸ s) (by simp only [inl_desc]) t
                                            /-
                                              🎉 no goals
                                            -/


theorem of_isBilimit {b : BinaryBicone X Y} (h : b.IsBilimit) :
    IsPushout (0 : 0 ⟶ X) (0 : 0 ⟶ Y) b.inl b.inr := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    b : CategoryTheory.Limits.BinaryBicone X Y
    h : b.IsBilimit
    ⊢ CategoryTheory.IsPushout 0 0 b.inl b.inr
  -/
  convert IsPushout.of_is_coproduct' h.isColimit HasZeroObject.zeroIsInitial
        /-
          case h.e'_7
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          X Y : C
          inst✝¹ : CategoryTheory.Limits.HasZeroObject C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          b : CategoryTheory.Limits.BinaryBicone X Y
          h : b.IsBilimit
          ⊢ Eq 0 (CategoryTheory.Limits.HasZeroObject.zeroIsInitial.to X)
        -/
        /-
          🎉 no goals
        -/
    <;> subsingleton
        /-
          🎉 no goals
        -/


@[simp]
theorem of_has_biproduct (X Y : C) [HasBinaryBiproduct X Y] :
    IsPushout (0 : 0 ⟶ X) (0 : 0 ⟶ Y) biprod.inl biprod.inr :=
  of_isBilimit (BinaryBiproduct.isBilimit X Y)


theorem inl_snd' {b : BinaryBicone X Y} (h : b.IsBilimit) :
    IsPushout b.inl (0 : X ⟶ 0) b.snd (0 : 0 ⟶ Y) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    b : CategoryTheory.Limits.BinaryBicone X Y
    h : b.IsBilimit
    ⊢ CategoryTheory.IsPushout b.inl 0 b.snd 0
  -/
  apply flip
  /-
    case h
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    b : CategoryTheory.Limits.BinaryBicone X Y
    h : b.IsBilimit
    ⊢ CategoryTheory.IsPushout 0 b.inl 0 b.snd
  -/
  refine of_left ?_ (by simp) (of_isBilimit h)
  /-
    case h
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    b : CategoryTheory.Limits.BinaryBicone X Y
    h : b.IsBilimit
    ⊢ CategoryTheory.IsPushout (CategoryTheory.CategoryStruct.comp 0 0) 0 0 (Categ …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The square
```
  X --inl--> X ⊞ Y
  |            |
  0           snd
  |            |
  v            v
  0 ---0-----> Y
```
is a pushout square.
-/
theorem inl_snd (X Y : C) [HasBinaryBiproduct X Y] :
    IsPushout biprod.inl (0 : X ⟶ 0) biprod.snd (0 : 0 ⟶ Y) :=
  inl_snd' (BinaryBiproduct.isBilimit X Y)


theorem inr_fst' {b : BinaryBicone X Y} (h : b.IsBilimit) :
    IsPushout b.inr (0 : Y ⟶ 0) b.fst (0 : 0 ⟶ X) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    b : CategoryTheory.Limits.BinaryBicone X Y
    h : b.IsBilimit
    ⊢ CategoryTheory.IsPushout b.inr 0 b.fst 0
  -/
  refine of_top ?_ (by simp) (of_isBilimit h)
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    b : CategoryTheory.Limits.BinaryBicone X Y
    h : b.IsBilimit
    ⊢ CategoryTheory.IsPushout 0 (CategoryTheory.CategoryStruct.comp 0 0) (Categor …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The square
```
  Y --inr--> X ⊞ Y
  |            |
  0           fst
  |            |
  v            v
  0 ---0-----> X
```
is a pushout square.
-/
theorem inr_fst (X Y : C) [HasBinaryBiproduct X Y] :
    IsPushout biprod.inr (0 : Y ⟶ 0) biprod.fst (0 : 0 ⟶ X) :=
  inr_fst' (BinaryBiproduct.isBilimit X Y)


theorem of_is_bilimit' {b : BinaryBicone X Y} (h : b.IsBilimit) :
    IsPushout b.fst b.snd (0 : X ⟶ 0) (0 : Y ⟶ 0) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    b : CategoryTheory.Limits.BinaryBicone X Y
    h : b.IsBilimit
    ⊢ CategoryTheory.IsPushout b.fst b.snd 0 0
  -/
  refine IsPushout.of_left ?_ (by simp) (IsPushout.inl_snd' h)
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    b : CategoryTheory.Limits.BinaryBicone X Y
    h : b.IsBilimit
    ⊢ CategoryTheory.IsPushout (CategoryTheory.CategoryStruct.comp b.inl b.fst) 0  …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem of_hasBinaryBiproduct (X Y : C) [HasBinaryBiproduct X Y] :
    IsPushout biprod.fst biprod.snd (0 : X ⟶ 0) (0 : Y ⟶ 0) :=
  of_is_bilimit' (BinaryBiproduct.isBilimit X Y)


instance hasPushout_biprod_fst_biprod_snd [HasBinaryBiproduct X Y] :
    HasPushout (biprod.fst : _ ⟶ X) (biprod.snd : _ ⟶ Y) :=
  HasColimit.mk ⟨_, (of_hasBinaryBiproduct X Y).isColimit⟩


/-- The pushout of `biprod.fst` and `biprod.snd` is the zero object. -/
def pushoutBiprodFstBiprodSnd [HasBinaryBiproduct X Y] :
    pushout (biprod.fst : _ ⟶ X) (biprod.snd : _ ⟶ Y) ≅ 0 :=
  colimit.isoColimitCocone ⟨_, (of_hasBinaryBiproduct X Y).isColimit⟩


theorem op (h : IsPushout f g inl inr) : IsPullback inr.op inl.op g.op f.op :=
  IsPullback.of_isLimit
    (IsLimit.ofIsoLimit
      (Limits.PushoutCocone.isColimitEquivIsLimitOp h.flip.cocone h.flip.isColimit)
      h.toCommSq.flip.coconeOp)


theorem unop {Z X Y P : Cᵒᵖ} {f : Z ⟶ X} {g : Z ⟶ Y} {inl : X ⟶ P} {inr : Y ⟶ P}
    (h : IsPushout f g inl inr) : IsPullback inr.unop inl.unop g.unop f.unop :=
  IsPullback.of_isLimit
    (IsLimit.ofIsoLimit
      (Limits.PushoutCocone.isColimitEquivIsLimitUnop h.flip.cocone h.flip.isColimit)
      h.toCommSq.flip.coconeUnop)


theorem of_horiz_isIso [IsIso f] [IsIso inr] (sq : CommSq f g inl inr) : IsPushout f g inl inr :=
  of_isColimit' sq
    (by
      refine
        PushoutCocone.IsColimit.mk _ (fun s => inv inr ≫ s.inr) (fun s => ?_)
          (by aesop_cat) (by aesop_cat)
      /-
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        Z X Y P : C
        f : Quiver.Hom Z X
        g : Quiver.Hom Z Y
        inl : Quiver.Hom X P
        inr : Quiver.Hom Y P
        inst✝¹ : CategoryTheory.IsIso f
        inst✝ : CategoryTheory.IsIso inr
        sq : CategoryTheory.CommSq f g inl inr
        s : CategoryTheory.Limits.PushoutCocone f g
        ⊢ Eq (CategoryTheory.CategoryStruct.comp inl ((fun s => CategoryTheory.Categor …
      -/
      simp only [← cancel_epi f, s.condition, sq.w_assoc, IsIso.hom_inv_id_assoc])
      /-
        🎉 no goals
      -/


theorem of_vert_isIso [IsIso g] [IsIso inl] (sq : CommSq f g inl inr) : IsPushout f g inl inr :=
  (of_horiz_isIso sq.flip).flip


                                                                            /-
                                                                              C : Type u₁
                                                                              inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                                                              Z X : C
                                                                              f : Quiver.Hom Z X
                                                                              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id Z)  …
                                                                            -/
lemma of_id_fst : IsPushout (𝟙 _) f f (𝟙 _) := IsPushout.of_horiz_isIso ⟨by simp⟩
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


                                                                           /-
                                                                             C : Type u₁
                                                                             inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                                                             Z X : C
                                                                             f : Quiver.Hom Z X
                                                                             ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.id X …
                                                                           -/
lemma of_id_snd : IsPushout f (𝟙 _) (𝟙 _) f := IsPushout.of_vert_isIso ⟨by simp⟩
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


/-- The following diagram is a pullback
```
X --f--> Z
|        |
id       id
v        v
X --f--> Z
```
-/
lemma id_vert (f : X ⟶ Z) : IsPushout f (𝟙 X) (𝟙 Z) f :=
                    /-
                      C : Type u₁
                      inst✝ : CategoryTheory.Category.{v₁, u₁} C
                      Z X : C
                      f : Quiver.Hom X Z
                      ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.id Z …
                    -/
  of_vert_isIso ⟨by simp only [Category.id_comp, Category.comp_id]⟩
                    /-
                      🎉 no goals
                    -/


/-- The following diagram is a pullback
```
X --id--> X
|         |
f         f
v         v
Z --id--> Z
```
-/
lemma id_horiz (f : X ⟶ Z) : IsPushout (𝟙 X) f f (𝟙 Z) :=
                     /-
                       C : Type u₁
                       inst✝ : CategoryTheory.Category.{v₁, u₁} C
                       Z X : C
                       f : Quiver.Hom X Z
                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id X)  …
                     -/
  of_horiz_isIso ⟨by simp only [Category.id_comp, Category.comp_id]⟩
                     /-
                       🎉 no goals
                     -/


/-- If `f : X ⟶ Y`, `g g' : Y ⟶ Z` forms a pullback square, then `f` is the equalizer of
`g` and `g'`. -/
noncomputable def IsPullback.isLimitFork (H : IsPullback f f g g') : IsLimit (Fork.ofι f H.w) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y Z : C
    f f' : Quiver.Hom X Y
    g g' : Quiver.Hom Y Z
    H : CategoryTheory.IsPullback f f g g'
    ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Fork.ofι f ⋯)
  -/
  fapply Fork.IsLimit.mk
    /-
      case lift
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X Y Z : C
      f f' : Quiver.Hom X Y
      g g' : Quiver.Hom Y Z
      H : CategoryTheory.IsPullback f f g g'
      ⊢ (s : CategoryTheory.Limits.Fork g g') → Quiver.Hom s.pt (CategoryTheory.Limi …
    -/
  · exact fun s => H.isLimit.lift (PullbackCone.mk s.ι s.ι s.condition)
    /-
      🎉 no goals
    -/
    /-
      case fac
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X Y Z : C
      f f' : Quiver.Hom X Y
      g g' : Quiver.Hom Y Z
      H : CategoryTheory.IsPullback f f g g'
      ⊢ ∀ (s : CategoryTheory.Limits.Fork g g'), Eq (CategoryTheory.CategoryStruct.c …
    -/
  · exact fun s => H.isLimit.fac _ WalkingCospan.left
    /-
      🎉 no goals
    -/
    /-
      case uniq
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X Y Z : C
      f f' : Quiver.Hom X Y
      g g' : Quiver.Hom Y Z
      H : CategoryTheory.IsPullback f f g g'
      ⊢ ∀ (s : CategoryTheory.Limits.Fork g g') (m : Quiver.Hom s.pt (CategoryTheory …
    -/
  · intro s m e
    /-
      case uniq
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X Y Z : C
      f f' : Quiver.Hom X Y
      g g' : Quiver.Hom Y Z
      H : CategoryTheory.IsPullback f f g g'
      s : CategoryTheory.Limits.Fork g g'
      m : Quiver.Hom s.pt (CategoryTheory.Limits.Fork.ofι f ⋯).pt
      e : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.Fork.ofι f …
      ⊢ Eq m (H.isLimit.lift (CategoryTheory.Limits.PullbackCone.mk s.ι s.ι ⋯))
    -/
    apply PullbackCone.IsLimit.hom_ext H.isLimit <;> refine e.trans ?_ <;> symm <;>
      /-
        case uniq.h₀
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        X Y Z : C
        f f' : Quiver.Hom X Y
        g g' : Quiver.Hom Y Z
        H : CategoryTheory.IsPullback f f g g'
        s : CategoryTheory.Limits.Fork g g'
        m : Quiver.Hom s.pt (CategoryTheory.Limits.Fork.ofι f ⋯).pt
        e : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.Fork.ofι f …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (H.isLimit.lift (CategoryTheory.Limit …
      -/
      /-
        🎉 no goals
      -/
      exact H.isLimit.fac _ _
      /-
        🎉 no goals
      -/


/-- If `f f' : X ⟶ Y`, `g : Y ⟶ Z` forms a pushout square, then `g` is the coequalizer of
`f` and `f'`. -/
noncomputable def IsPushout.isLimitFork (H : IsPushout f f' g g) :
    IsColimit (Cofork.ofπ g H.w) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y Z : C
    f f' : Quiver.Hom X Y
    g g' : Quiver.Hom Y Z
    H : CategoryTheory.IsPushout f f' g g
    ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofork.ofπ g ⋯)
  -/
  fapply Cofork.IsColimit.mk
    /-
      case desc
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X Y Z : C
      f f' : Quiver.Hom X Y
      g g' : Quiver.Hom Y Z
      H : CategoryTheory.IsPushout f f' g g
      ⊢ (s : CategoryTheory.Limits.Cofork f f') → Quiver.Hom (CategoryTheory.Limits. …
    -/
  · exact fun s => H.isColimit.desc (PushoutCocone.mk s.π s.π s.condition)
    /-
      🎉 no goals
    -/
    /-
      case fac
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X Y Z : C
      f f' : Quiver.Hom X Y
      g g' : Quiver.Hom Y Z
      H : CategoryTheory.IsPushout f f' g g
      ⊢ ∀ (s : CategoryTheory.Limits.Cofork f f'), Eq (CategoryTheory.CategoryStruct …
    -/
  · exact fun s => H.isColimit.fac _ WalkingSpan.left
    /-
      🎉 no goals
    -/
    /-
      case uniq
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X Y Z : C
      f f' : Quiver.Hom X Y
      g g' : Quiver.Hom Y Z
      H : CategoryTheory.IsPushout f f' g g
      ⊢ ∀ (s : CategoryTheory.Limits.Cofork f f') (m : Quiver.Hom (CategoryTheory.Li …
    -/
  · intro s m e
    /-
      case uniq
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X Y Z : C
      f f' : Quiver.Hom X Y
      g g' : Quiver.Hom Y Z
      H : CategoryTheory.IsPushout f f' g g
      s : CategoryTheory.Limits.Cofork f f'
      m : Quiver.Hom (CategoryTheory.Limits.Cofork.ofπ g ⋯).pt s.pt
      e : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.ofπ g …
      ⊢ Eq m (H.isColimit.desc (CategoryTheory.Limits.PushoutCocone.mk s.π s.π ⋯))
    -/
    apply PushoutCocone.IsColimit.hom_ext H.isColimit <;> refine e.trans ?_ <;> symm <;>
      /-
        case uniq.h₀
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        X Y Z : C
        f f' : Quiver.Hom X Y
        g g' : Quiver.Hom Y Z
        H : CategoryTheory.IsPushout f f' g g
        s : CategoryTheory.Limits.Cofork f f'
        m : Quiver.Hom (CategoryTheory.Limits.Cofork.ofπ g ⋯).pt s.pt
        e : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.ofπ g …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp H.cocone.inl (H.isColimit.desc (Categ …
      -/
      /-
        🎉 no goals
      -/
      exact H.isColimit.fac _ _
      /-
        🎉 no goals
      -/


theorem of_isPullback_isPushout (p₁ : IsPullback f g h i) (p₂ : IsPushout f g h i) :
    BicartesianSq f g h i :=
  BicartesianSq.mk p₁ p₂.isColimit'


theorem flip (p : BicartesianSq f g h i) : BicartesianSq g f i h :=
  of_isPullback_isPushout p.toIsPullback.flip p.toIsPushout.flip


/-- ```
 X ⊞ Y --fst--> X
   |            |
  snd           0
   |            |
   v            v
   Y -----0---> 0
```
is a bicartesian square.
-/
theorem of_is_biproduct₁ {b : BinaryBicone X Y} (h : b.IsBilimit) :
    BicartesianSq b.fst b.snd (0 : X ⟶ 0) (0 : Y ⟶ 0) :=
  of_isPullback_isPushout (IsPullback.of_isBilimit h) (IsPushout.of_is_bilimit' h)


/-- ```
   0 -----0---> X
   |            |
   0           inl
   |            |
   v            v
   Y --inr--> X ⊞ Y
```
is a bicartesian square.
-/
theorem of_is_biproduct₂ {b : BinaryBicone X Y} (h : b.IsBilimit) :
    BicartesianSq (0 : 0 ⟶ X) (0 : 0 ⟶ Y) b.inl b.inr :=
  of_isPullback_isPushout (IsPullback.of_is_bilimit' h) (IsPushout.of_isBilimit h)


/-- ```
 X ⊞ Y --fst--> X
   |            |
  snd           0
   |            |
   v            v
   Y -----0---> 0
```
is a bicartesian square.
-/
@[simp]
theorem of_has_biproduct₁ [HasBinaryBiproduct X Y] :
    BicartesianSq biprod.fst biprod.snd (0 : X ⟶ 0) (0 : Y ⟶ 0) := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    inst✝² : CategoryTheory.Limits.HasZeroObject C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X Y
    ⊢ CategoryTheory.BicartesianSq CategoryTheory.Limits.biprod.fst CategoryTheory …
  -/
  convert of_is_biproduct₁ (BinaryBiproduct.isBilimit X Y)
  /-
    🎉 no goals
  -/


/-- ```
   0 -----0---> X
   |            |
   0           inl
   |            |
   v            v
   Y --inr--> X ⊞ Y
```
is a bicartesian square.
-/
@[simp]
theorem of_has_biproduct₂ [HasBinaryBiproduct X Y] :
    BicartesianSq (0 : 0 ⟶ X) (0 : 0 ⟶ Y) biprod.inl biprod.inr := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    inst✝² : CategoryTheory.Limits.HasZeroObject C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X Y
    ⊢ CategoryTheory.BicartesianSq 0 0 CategoryTheory.Limits.biprod.inl CategoryTh …
  -/
  convert of_is_biproduct₂ (BinaryBiproduct.isBilimit X Y)
  /-
    🎉 no goals
  -/


theorem Functor.map_isPullback [PreservesLimit (cospan h i) F] (s : IsPullback f g h i) :
    IsPullback (F.map f) (F.map g) (F.map h) (F.map i) := by
  -- This is made slightly awkward because `C` and `D` have different universes,
  -- and so the relevant `WalkingCospan` diagrams live in different universes too!
  refine
    IsPullback.of_isLimit' (F.map_commSq s.toCommSq)
      (IsLimit.equivOfNatIsoOfIso (cospanCompIso F h i) _ _ (WalkingCospan.ext ?_ ?_ ?_)
        (isLimitOfPreserves F s.isLimit))
    /-
      case refine_1
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      W X Y Z : C
      f : Quiver.Hom W X
      g : Quiver.Hom W Y
      h : Quiver.Hom X Z
      i : Quiver.Hom Y Z
      inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.cospan h i …
      s : CategoryTheory.IsPullback f g h i
      ⊢ CategoryTheory.Iso ((CategoryTheory.Limits.Cones.postcompose (CategoryTheory …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      W X Y Z : C
      f : Quiver.Hom W X
      g : Quiver.Hom W Y
      h : Quiver.Hom X Z
      i : Quiver.Hom Y Z
      inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.cospan h i …
      s : CategoryTheory.IsPullback f g h i
      ⊢ Eq (((CategoryTheory.Limits.Cones.postcompose (CategoryTheory.Limits.cospanC …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      W X Y Z : C
      f : Quiver.Hom W X
      g : Quiver.Hom W Y
      h : Quiver.Hom X Z
      i : Quiver.Hom Y Z
      inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.cospan h i …
      s : CategoryTheory.IsPullback f g h i
      ⊢ Eq (((CategoryTheory.Limits.Cones.postcompose (CategoryTheory.Limits.cospanC …
    -/
  · simp
    /-
      🎉 no goals
    -/


theorem Functor.map_isPushout [PreservesColimit (span f g) F] (s : IsPushout f g h i) :
    IsPushout (F.map f) (F.map g) (F.map h) (F.map i) := by
  refine
    IsPushout.of_isColimit' (F.map_commSq s.toCommSq)
      (IsColimit.equivOfNatIsoOfIso (spanCompIso F f g) _ _ (WalkingSpan.ext ?_ ?_ ?_)
        (isColimitOfPreserves F s.isColimit))
    /-
      case refine_1
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      W X Y Z : C
      f : Quiver.Hom W X
      g : Quiver.Hom W Y
      h : Quiver.Hom X Z
      i : Quiver.Hom Y Z
      inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.span f g …
      s : CategoryTheory.IsPushout f g h i
      ⊢ CategoryTheory.Iso ((CategoryTheory.Limits.Cocones.precompose (CategoryTheor …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      W X Y Z : C
      f : Quiver.Hom W X
      g : Quiver.Hom W Y
      h : Quiver.Hom X Z
      i : Quiver.Hom Y Z
      inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.span f g …
      s : CategoryTheory.IsPushout f g h i
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Limits.Cocones.prec …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      W X Y Z : C
      f : Quiver.Hom W X
      g : Quiver.Hom W Y
      h : Quiver.Hom X Z
      i : Quiver.Hom Y Z
      inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.span f g …
      s : CategoryTheory.IsPushout f g h i
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Limits.Cocones.prec …
    -/
  · simp
    /-
      🎉 no goals
    -/


alias IsPullback.map := Functor.map_isPullback


alias IsPushout.map := Functor.map_isPushout


theorem IsPullback.of_map [ReflectsLimit (cospan h i) F] (e : f ≫ h = g ≫ i)
    (H : IsPullback (F.map f) (F.map g) (F.map h) (F.map i)) : IsPullback f g h i := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    W X Y Z : C
    f : Quiver.Hom W X
    g : Quiver.Hom W Y
    h : Quiver.Hom X Z
    i : Quiver.Hom Y Z
    inst✝ : CategoryTheory.Limits.ReflectsLimit (CategoryTheory.Limits.cospan h i) F
    e : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStruct …
    H : CategoryTheory.IsPullback (F.map f) (F.map g) (F.map h) (F.map i)
    ⊢ CategoryTheory.IsPullback f g h i
  -/
  refine ⟨⟨e⟩, ⟨isLimitOfReflects F <| ?_⟩⟩
  refine
    (IsLimit.equivOfNatIsoOfIso (cospanCompIso F h i) _ _ (WalkingCospan.ext ?_ ?_ ?_)).symm
      H.isLimit
  exacts [Iso.refl _, (Category.comp_id _).trans (Category.id_comp _).symm,
    (Category.comp_id _).trans (Category.id_comp _).symm]


theorem IsPullback.of_map_of_faithful [ReflectsLimit (cospan h i) F] [F.Faithful]
    (H : IsPullback (F.map f) (F.map g) (F.map h) (F.map i)) : IsPullback f g h i :=
                                    /-
                                      C : Type u₁
                                      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                                      D : Type u₂
                                      inst✝² : CategoryTheory.Category.{v₂, u₂} D
                                      F : CategoryTheory.Functor C D
                                      W X Y Z : C
                                      f : Quiver.Hom W X
                                      g : Quiver.Hom W Y
                                      h : Quiver.Hom X Z
                                      i : Quiver.Hom Y Z
                                      inst✝¹ : CategoryTheory.Limits.ReflectsLimit (CategoryTheory.Limits.cospan h i …
                                      inst✝ : F.Faithful
                                      H : CategoryTheory.IsPullback (F.map f) (F.map g) (F.map h) (F.map i)
                                      ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp f h)) (F.map (CategoryTheory.C …
                                    -/
  H.of_map F (F.map_injective <| by simpa only [F.map_comp] using H.w)
                                    /-
                                      🎉 no goals
                                    -/


theorem IsPullback.map_iff {D : Type*} [Category D] (F : C ⥤ D) [PreservesLimit (cospan h i) F]
    [ReflectsLimit (cospan h i) F] (e : f ≫ h = g ≫ i) :
    IsPullback (F.map f) (F.map g) (F.map h) (F.map i) ↔ IsPullback f g h i :=
  ⟨fun h => h.of_map F e, fun h => h.map F⟩


theorem IsPushout.of_map [ReflectsColimit (span f g) F] (e : f ≫ h = g ≫ i)
    (H : IsPushout (F.map f) (F.map g) (F.map h) (F.map i)) : IsPushout f g h i := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    W X Y Z : C
    f : Quiver.Hom W X
    g : Quiver.Hom W Y
    h : Quiver.Hom X Z
    i : Quiver.Hom Y Z
    inst✝ : CategoryTheory.Limits.ReflectsColimit (CategoryTheory.Limits.span f g) F
    e : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStruct …
    H : CategoryTheory.IsPushout (F.map f) (F.map g) (F.map h) (F.map i)
    ⊢ CategoryTheory.IsPushout f g h i
  -/
  refine ⟨⟨e⟩, ⟨isColimitOfReflects F <| ?_⟩⟩
  refine
    (IsColimit.equivOfNatIsoOfIso (spanCompIso F f g) _ _ (WalkingSpan.ext ?_ ?_ ?_)).symm
      H.isColimit
  exacts [Iso.refl _, (Category.comp_id _).trans (Category.id_comp _),
    (Category.comp_id _).trans (Category.id_comp _)]


theorem IsPushout.of_map_of_faithful [ReflectsColimit (span f g) F] [F.Faithful]
    (H : IsPushout (F.map f) (F.map g) (F.map h) (F.map i)) : IsPushout f g h i :=
                                    /-
                                      C : Type u₁
                                      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                                      D : Type u₂
                                      inst✝² : CategoryTheory.Category.{v₂, u₂} D
                                      F : CategoryTheory.Functor C D
                                      W X Y Z : C
                                      f : Quiver.Hom W X
                                      g : Quiver.Hom W Y
                                      h : Quiver.Hom X Z
                                      i : Quiver.Hom Y Z
                                      inst✝¹ : CategoryTheory.Limits.ReflectsColimit (CategoryTheory.Limits.span f g …
                                      inst✝ : F.Faithful
                                      H : CategoryTheory.IsPushout (F.map f) (F.map g) (F.map h) (F.map i)
                                      ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp f h)) (F.map (CategoryTheory.C …
                                    -/
  H.of_map F (F.map_injective <| by simpa only [F.map_comp] using H.w)
                                    /-
                                      🎉 no goals
                                    -/


theorem IsPushout.map_iff {D : Type*} [Category D] (F : C ⥤ D) [PreservesColimit (span f g) F]
    [ReflectsColimit (span f g) F] (e : f ≫ h = g ≫ i) :
    IsPushout (F.map f) (F.map g) (F.map h) (F.map i) ↔ IsPushout f g h i :=
  ⟨fun h => h.of_map F e, fun h => h.map F⟩


