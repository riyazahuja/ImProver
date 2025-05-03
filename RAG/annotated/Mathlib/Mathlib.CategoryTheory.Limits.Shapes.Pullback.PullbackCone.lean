/-- A pullback cone is just a cone on the cospan formed by two morphisms `f : X ⟶ Z` and
    `g : Y ⟶ Z`. -/
abbrev PullbackCone (f : X ⟶ Z) (g : Y ⟶ Z) :=
  Cone (cospan f g)


/-- The first projection of a pullback cone. -/
abbrev fst (t : PullbackCone f g) : t.pt ⟶ X :=
  t.π.app WalkingCospan.left


/-- The second projection of a pullback cone. -/
abbrev snd (t : PullbackCone f g) : t.pt ⟶ Y :=
  t.π.app WalkingCospan.right


@[simp]
theorem π_app_left (c : PullbackCone f g) : c.π.app WalkingCospan.left = c.fst := rfl


@[simp]
theorem π_app_right (c : PullbackCone f g) : c.π.app WalkingCospan.right = c.snd := rfl


@[simp]
theorem condition_one (t : PullbackCone f g) : t.π.app WalkingCospan.one = t.fst ≫ f := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    t : CategoryTheory.Limits.PullbackCone f g
    ⊢ Eq (t.π.app CategoryTheory.Limits.WalkingCospan.one) (CategoryTheory.Categor …
  -/
  have w := t.π.naturality WalkingCospan.Hom.inl
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    t : CategoryTheory.Limits.PullbackCone f g
    w : Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const Cat …
    ⊢ Eq (t.π.app CategoryTheory.Limits.WalkingCospan.one) (CategoryTheory.Categor …
  -/
  dsimp at w; simpa using w
              /-
                🎉 no goals
              -/


/-- A pullback cone on `f` and `g` is determined by morphisms `fst : W ⟶ X` and `snd : W ⟶ Y`
    such that `fst ≫ f = snd ≫ g`. -/
@[simps]
def mk {W : C} (fst : W ⟶ X) (snd : W ⟶ Y) (eq : fst ≫ f = snd ≫ g) : PullbackCone f g where
  pt := W
  π := { app := fun j => Option.casesOn j (fst ≫ f) fun j' => WalkingPair.casesOn j' fst snd
                          /-
                            C : Type u
                            inst✝ : CategoryTheory.Category.{v, u} C
                            W✝ X Y Z : C
                            f : Quiver.Hom X Z
                            g : Quiver.Hom Y Z
                            W : C
                            fst : Quiver.Hom W X
                            snd : Quiver.Hom W Y
                            eq : Eq (CategoryTheory.CategoryStruct.comp fst f) (CategoryTheory.CategoryStr …
                            ⊢ ∀ ⦃X_1 Y_1 : CategoryTheory.Limits.WalkingCospan⦄ (f_1 : Quiver.Hom X_1 Y_1) …
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
                                                               /-
                                                                 🎉 no goals
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
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/
         naturality := by rintro (⟨⟩ | ⟨⟨⟩⟩) (⟨⟩ | ⟨⟨⟩⟩) j <;> cases j <;> dsimp <;> simp [eq] }
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


@[simp]
theorem mk_π_app_left {W : C} (fst : W ⟶ X) (snd : W ⟶ Y) (eq : fst ≫ f = snd ≫ g) :
    (mk fst snd eq).π.app WalkingCospan.left = fst := rfl


@[simp]
theorem mk_π_app_right {W : C} (fst : W ⟶ X) (snd : W ⟶ Y) (eq : fst ≫ f = snd ≫ g) :
    (mk fst snd eq).π.app WalkingCospan.right = snd := rfl


@[simp]
theorem mk_π_app_one {W : C} (fst : W ⟶ X) (snd : W ⟶ Y) (eq : fst ≫ f = snd ≫ g) :
    (mk fst snd eq).π.app WalkingCospan.one = fst ≫ f := rfl


@[simp]
theorem mk_fst {W : C} (fst : W ⟶ X) (snd : W ⟶ Y) (eq : fst ≫ f = snd ≫ g) :
    (mk fst snd eq).fst = fst := rfl


@[simp]
theorem mk_snd {W : C} (fst : W ⟶ X) (snd : W ⟶ Y) (eq : fst ≫ f = snd ≫ g) :
    (mk fst snd eq).snd = snd := rfl


@[reassoc]
theorem condition (t : PullbackCone f g) : fst t ≫ f = snd t ≫ g :=
  (t.w inl).trans (t.w inr).symm


/-- To check whether two morphisms are equalized by the maps of a pullback cone, it suffices to
check it for `fst t` and `snd t` -/
theorem equalizer_ext (t : PullbackCone f g) {W : C} {k l : W ⟶ t.pt} (h₀ : k ≫ fst t = l ≫ fst t)
    (h₁ : k ≫ snd t = l ≫ snd t) : ∀ j : WalkingCospan, k ≫ t.π.app j = l ≫ t.π.app j
  | some WalkingPair.left => h₀
  | some WalkingPair.right => h₁
               /-
                 C : Type u
                 inst✝ : CategoryTheory.Category.{v, u} C
                 X Y Z : C
                 f : Quiver.Hom X Z
                 g : Quiver.Hom Y Z
                 t : CategoryTheory.Limits.PullbackCone f g
                 W : C
                 k l : Quiver.Hom W t.pt
                 h₀ : Eq (CategoryTheory.CategoryStruct.comp k t.fst) (CategoryTheory.CategoryS …
                 h₁ : Eq (CategoryTheory.CategoryStruct.comp k t.snd) (CategoryTheory.CategoryS …
                 ⊢ Eq (CategoryTheory.CategoryStruct.comp k (t.π.app Option.none)) (CategoryThe …
               -/
  | none => by rw [← t.w inl, reassoc_of% h₀]
               /-
                 🎉 no goals
               -/


/-- To construct an isomorphism of pullback cones, it suffices to construct an isomorphism
of the cone points and check it commutes with `fst` and `snd`. -/
def ext {s t : PullbackCone f g} (i : s.pt ≅ t.pt) (w₁ : s.fst = i.hom ≫ t.fst := by aesop_cat)
    (w₂ : s.snd = i.hom ≫ t.snd := by aesop_cat) : s ≅ t :=
  WalkingCospan.ext i w₁ w₂


/-- The natural isomorphism between a pullback cone and the corresponding pullback cone
reconstructed using `PullbackCone.mk`. -/
@[simps!]
def eta (t : PullbackCone f g) : t ≅ mk t.fst t.snd t.condition :=
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    W X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    t : CategoryTheory.Limits.PullbackCone f g
    ⊢ Eq t.fst (CategoryTheory.CategoryStruct.comp (CategoryTheory.Iso.refl t.pt). …
  -/
  /-
    🎉 no goals
  -/
  PullbackCone.ext (Iso.refl _)
  /-
    🎉 no goals
  -/


/-- This is a slightly more convenient method to verify that a pullback cone is a limit cone. It
    only asks for a proof of facts that carry any mathematical content -/
def isLimitAux (t : PullbackCone f g) (lift : ∀ s : PullbackCone f g, s.pt ⟶ t.pt)
    (fac_left : ∀ s : PullbackCone f g, lift s ≫ t.fst = s.fst)
    (fac_right : ∀ s : PullbackCone f g, lift s ≫ t.snd = s.snd)
    (uniq : ∀ (s : PullbackCone f g) (m : s.pt ⟶ t.pt)
      (_ : ∀ j : WalkingCospan, m ≫ t.π.app j = s.π.app j), m = lift s) : IsLimit t :=
  { lift
    fac := fun s j => Option.casesOn j (by
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          W X Y Z : C
          f : Quiver.Hom X Z
          g : Quiver.Hom Y Z
          t : CategoryTheory.Limits.PullbackCone f g
          lift : (s : CategoryTheory.Limits.PullbackCone f g) → Quiver.Hom s.pt t.pt
          fac_left : ∀ (s : CategoryTheory.Limits.PullbackCone f g), Eq (CategoryTheory. …
          fac_right : ∀ (s : CategoryTheory.Limits.PullbackCone f g), Eq (CategoryTheory …
          uniq : ∀ (s : CategoryTheory.Limits.PullbackCone f g) (m : Quiver.Hom s.pt t.p …
          s : CategoryTheory.Limits.Cone (CategoryTheory.Limits.cospan f g)
          j : CategoryTheory.Limits.WalkingCospan
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (lift s) (t.π.app Option.none)) (s.π. …
        -/
        rw [← s.w inl, ← t.w inl, ← Category.assoc]
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          W X Y Z : C
          f : Quiver.Hom X Z
          g : Quiver.Hom Y Z
          t : CategoryTheory.Limits.PullbackCone f g
          lift : (s : CategoryTheory.Limits.PullbackCone f g) → Quiver.Hom s.pt t.pt
          fac_left : ∀ (s : CategoryTheory.Limits.PullbackCone f g), Eq (CategoryTheory. …
          fac_right : ∀ (s : CategoryTheory.Limits.PullbackCone f g), Eq (CategoryTheory …
          uniq : ∀ (s : CategoryTheory.Limits.PullbackCone f g) (m : Quiver.Hom s.pt t.p …
          s : CategoryTheory.Limits.Cone (CategoryTheory.Limits.cospan f g)
          j : CategoryTheory.Limits.WalkingCospan
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        congr
        /-
          case e_a
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          W X Y Z : C
          f : Quiver.Hom X Z
          g : Quiver.Hom Y Z
          t : CategoryTheory.Limits.PullbackCone f g
          lift : (s : CategoryTheory.Limits.PullbackCone f g) → Quiver.Hom s.pt t.pt
          fac_left : ∀ (s : CategoryTheory.Limits.PullbackCone f g), Eq (CategoryTheory. …
          fac_right : ∀ (s : CategoryTheory.Limits.PullbackCone f g), Eq (CategoryTheory …
          uniq : ∀ (s : CategoryTheory.Limits.PullbackCone f g) (m : Quiver.Hom s.pt t.p …
          s : CategoryTheory.Limits.Cone (CategoryTheory.Limits.cospan f g)
          j : CategoryTheory.Limits.WalkingCospan
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (lift s) (t.π.app CategoryTheory.Limi …
        -/
        exact fac_left s)
        /-
          🎉 no goals
        -/
      fun j' => WalkingPair.casesOn j' (fac_left s) (fac_right s)
    uniq := uniq }


/-- This is another convenient method to verify that a pullback cone is a limit cone. It
    only asks for a proof of facts that carry any mathematical content, and allows access to the
    same `s` for all parts. -/
def isLimitAux' (t : PullbackCone f g)
    (create :
      ∀ s : PullbackCone f g,
        { l //
          l ≫ t.fst = s.fst ∧
            l ≫ t.snd = s.snd ∧ ∀ {m}, m ≫ t.fst = s.fst → m ≫ t.snd = s.snd → m = l }) :
    Limits.IsLimit t :=
  PullbackCone.isLimitAux t (fun s => (create s).1) (fun s => (create s).2.1)
    (fun s => (create s).2.2.1) fun s _ w =>
    (create s).2.2.2 (w WalkingCospan.left) (w WalkingCospan.right)


/-- This is a more convenient formulation to show that a `PullbackCone` constructed using
`PullbackCone.mk` is a limit cone.
-/
def IsLimit.mk {W : C} {fst : W ⟶ X} {snd : W ⟶ Y} (eq : fst ≫ f = snd ≫ g)
    (lift : ∀ s : PullbackCone f g, s.pt ⟶ W)
    (fac_left : ∀ s : PullbackCone f g, lift s ≫ fst = s.fst)
    (fac_right : ∀ s : PullbackCone f g, lift s ≫ snd = s.snd)
    (uniq :
      ∀ (s : PullbackCone f g) (m : s.pt ⟶ W) (_ : m ≫ fst = s.fst) (_ : m ≫ snd = s.snd),
        m = lift s) :
    IsLimit (mk fst snd eq) :=
  isLimitAux _ lift fac_left fac_right fun s m w =>
    uniq s m (w WalkingCospan.left) (w WalkingCospan.right)


theorem IsLimit.hom_ext {t : PullbackCone f g} (ht : IsLimit t) {W : C} {k l : W ⟶ t.pt}
    (h₀ : k ≫ fst t = l ≫ fst t) (h₁ : k ≫ snd t = l ≫ snd t) : k = l :=
  ht.hom_ext <| equalizer_ext _ h₀ h₁

-- Porting note: `IsLimit.lift` and the two following simp lemmas were introduced to ease the port

/-- If `t` is a limit pullback cone over `f` and `g` and `h : W ⟶ X` and `k : W ⟶ Y` are such that
    `h ≫ f = k ≫ g`, then we get `l : W ⟶ t.pt`, which satisfies `l ≫ fst t = h`
    and `l ≫ snd t = k`, see `IsLimit.lift_fst` and `IsLimit.lift_snd`. -/
def IsLimit.lift {t : PullbackCone f g} (ht : IsLimit t) {W : C} (h : W ⟶ X) (k : W ⟶ Y)
    (w : h ≫ f = k ≫ g) : W ⟶ t.pt :=
  ht.lift <| PullbackCone.mk _ _ w


@[reassoc (attr := simp)]
lemma IsLimit.lift_fst {t : PullbackCone f g} (ht : IsLimit t) {W : C} (h : W ⟶ X) (k : W ⟶ Y)
    (w : h ≫ f = k ≫ g) : IsLimit.lift ht h k w ≫ fst t = h := ht.fac _ _


@[reassoc (attr := simp)]
lemma IsLimit.lift_snd {t : PullbackCone f g} (ht : IsLimit t) {W : C} (h : W ⟶ X) (k : W ⟶ Y)
    (w : h ≫ f = k ≫ g) : IsLimit.lift ht h k w ≫ snd t = k := ht.fac _ _


/-- If `t` is a limit pullback cone over `f` and `g` and `h : W ⟶ X` and `k : W ⟶ Y` are such that
    `h ≫ f = k ≫ g`, then we have `l : W ⟶ t.pt` satisfying `l ≫ fst t = h` and `l ≫ snd t = k`.
    -/
def IsLimit.lift' {t : PullbackCone f g} (ht : IsLimit t) {W : C} (h : W ⟶ X) (k : W ⟶ Y)
    (w : h ≫ f = k ≫ g) : { l : W ⟶ t.pt // l ≫ fst t = h ∧ l ≫ snd t = k } :=
                             /-
                               C : Type u
                               inst✝ : CategoryTheory.Category.{v, u} C
                               W✝ X Y Z : C
                               f : Quiver.Hom X Z
                               g : Quiver.Hom Y Z
                               t : CategoryTheory.Limits.PullbackCone f g
                               ht : CategoryTheory.Limits.IsLimit t
                               W : C
                               h : Quiver.Hom W X
                               k : Quiver.Hom W Y
                               w : Eq (CategoryTheory.CategoryStruct.comp h f) (CategoryTheory.CategoryStruct …
                               ⊢ And (Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PullbackC …
                             -/
  ⟨IsLimit.lift ht h k w, by simp⟩
                             /-
                               🎉 no goals
                             -/


/-- The pullback cone reconstructed using `PullbackCone.mk` from a pullback cone that is a
limit, is also a limit. -/
def mkSelfIsLimit {t : PullbackCone f g} (ht : IsLimit t) : IsLimit (mk t.fst t.snd t.condition) :=
  IsLimit.ofIsoLimit ht (eta t)


/-- The pullback cone obtained by flipping `fst` and `snd`. -/
def flip : PullbackCone g f := PullbackCone.mk _ _ t.condition.symm


@[simp] lemma flip_pt : t.flip.pt = t.pt := rfl

@[simp] lemma flip_fst : t.flip.fst = t.snd := rfl

@[simp] lemma flip_snd : t.flip.snd = t.fst := rfl


/-- Flipping a pullback cone twice gives an isomorphic cone. -/
                                                                       /-
                                                                         C : Type u
                                                                         inst✝ : CategoryTheory.Category.{v, u} C
                                                                         W X Y Z : C
                                                                         f : Quiver.Hom X Z
                                                                         g : Quiver.Hom Y Z
                                                                         t : CategoryTheory.Limits.PullbackCone f g
                                                                         ⊢ Eq t.flip.flip.fst (CategoryTheory.CategoryStruct.comp (CategoryTheory.Iso.r …
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
def flipFlipIso : t.flip.flip ≅ t := PullbackCone.ext (Iso.refl _) (by simp) (by simp)
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


/-- The flip of a pullback square is a pullback square. -/
def flipIsLimit (ht : IsLimit t) : IsLimit t.flip :=
                                             /-
                                               C : Type u
                                               inst✝ : CategoryTheory.Category.{v, u} C
                                               W X Y Z : C
                                               f : Quiver.Hom X Z
                                               g : Quiver.Hom Y Z
                                               t : CategoryTheory.Limits.PullbackCone f g
                                               ht : CategoryTheory.Limits.IsLimit t
                                               ⊢ ∀ (s : CategoryTheory.Limits.PullbackCone g f), Eq (CategoryTheory.CategoryS …
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
  IsLimit.mk _ (fun s => ht.lift s.flip) (by simp) (by simp) (fun s m h₁ h₂ => by
                                                       /-
                                                         🎉 no goals
                                                       -/
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      W X Y Z : C
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      t : CategoryTheory.Limits.PullbackCone f g
      ht : CategoryTheory.Limits.IsLimit t
      s : CategoryTheory.Limits.PullbackCone g f
      m : Quiver.Hom s.pt t.pt
      h₁ : Eq (CategoryTheory.CategoryStruct.comp m t.snd) s.fst
      h₂ : Eq (CategoryTheory.CategoryStruct.comp m t.fst) s.snd
      ⊢ Eq m ((fun s => ht.lift s.flip) s)
    -/
                                 /-
                                   🎉 no goals
                                 -/
    apply IsLimit.hom_ext ht <;> simp [h₁, h₂])
                                 /-
                                   🎉 no goals
                                 -/


/-- A square is a pullback square if its flip is. -/
def isLimitOfFlip (ht : IsLimit t.flip) : IsLimit t :=
  IsLimit.ofIsoLimit (flipIsLimit ht) t.flipFlipIso


/-- This is a helper construction that can be useful when verifying that a category has all
    pullbacks. Given `F : WalkingCospan ⥤ C`, which is really the same as
    `cospan (F.map inl) (F.map inr)`, and a pullback cone on `F.map inl` and `F.map inr`, we
    get a cone on `F`.

    If you're thinking about using this, have a look at `hasPullbacks_of_hasLimit_cospan`,
    which you may find to be an easier way of achieving your goal. -/
@[simps]
def Cone.ofPullbackCone {F : WalkingCospan ⥤ C} (t : PullbackCone (F.map inl) (F.map inr)) :
    Cone F where
  pt := t.pt
  π := t.π ≫ (diagramIsoCospan F).inv


/-- Given `F : WalkingCospan ⥤ C`, which is really the same as `cospan (F.map inl) (F.map inr)`,
    and a cone on `F`, we get a pullback cone on `F.map inl` and `F.map inr`. -/
@[simps]
def PullbackCone.ofCone {F : WalkingCospan ⥤ C} (t : Cone F) :
    PullbackCone (F.map inl) (F.map inr) where
  pt := t.pt
  π := t.π ≫ (diagramIsoCospan F).hom


/-- A diagram `WalkingCospan ⥤ C` is isomorphic to some `PullbackCone.mk` after
composing with `diagramIsoCospan`. -/
@[simps!]
def PullbackCone.isoMk {F : WalkingCospan ⥤ C} (t : Cone F) :
    (Cones.postcompose (diagramIsoCospan.{v} _).hom).obj t ≅
      PullbackCone.mk (t.π.app WalkingCospan.left) (t.π.app WalkingCospan.right)
        ((t.π.naturality inl).symm.trans (t.π.naturality inr : _)) :=
  Cones.ext (Iso.refl _) <| by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      W X Y Z : C
      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingCospan C
      t : CategoryTheory.Limits.Cone F
      ⊢ ∀ (j : CategoryTheory.Limits.WalkingCospan), Eq (((CategoryTheory.Limits.Con …
    -/
    rintro (_ | (_ | _)) <;>
        /-
          case none
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          W X Y Z : C
          F : CategoryTheory.Functor CategoryTheory.Limits.WalkingCospan C
          t : CategoryTheory.Limits.Cone F
          ⊢ Eq (((CategoryTheory.Limits.Cones.postcompose (CategoryTheory.Limits.diagram …
        -/
        /-
          case none
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          W X Y Z : C
          F : CategoryTheory.Functor CategoryTheory.Limits.WalkingCospan C
          t : CategoryTheory.Limits.Cone F
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (t.π.app Option.none) (CategoryTheory …
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
        /-
          case some.right
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          W X Y Z : C
          F : CategoryTheory.Functor CategoryTheory.Limits.WalkingCospan C
          t : CategoryTheory.Limits.Cone F
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (t.π.app (Option.some CategoryTheory. …
        -/
        simp
        /-
          🎉 no goals
        -/


/-- A pushout cocone is just a cocone on the span formed by two morphisms `f : X ⟶ Y` and
    `g : X ⟶ Z`. -/
abbrev PushoutCocone (f : X ⟶ Y) (g : X ⟶ Z) :=
  Cocone (span f g)


/-- The first inclusion of a pushout cocone. -/
abbrev inl (t : PushoutCocone f g) : Y ⟶ t.pt :=
  t.ι.app WalkingSpan.left


/-- The second inclusion of a pushout cocone. -/
abbrev inr (t : PushoutCocone f g) : Z ⟶ t.pt :=
  t.ι.app WalkingSpan.right


@[simp]
theorem ι_app_left (c : PushoutCocone f g) : c.ι.app WalkingSpan.left = c.inl := rfl


@[simp]
theorem ι_app_right (c : PushoutCocone f g) : c.ι.app WalkingSpan.right = c.inr := rfl


@[simp]
theorem condition_zero (t : PushoutCocone f g) : t.ι.app WalkingSpan.zero = f ≫ t.inl := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom X Z
    t : CategoryTheory.Limits.PushoutCocone f g
    ⊢ Eq (t.ι.app CategoryTheory.Limits.WalkingSpan.zero) (CategoryTheory.Category …
  -/
  have w := t.ι.naturality WalkingSpan.Hom.fst
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom X Z
    t : CategoryTheory.Limits.PushoutCocone f g
    w : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.span f g).m …
    ⊢ Eq (t.ι.app CategoryTheory.Limits.WalkingSpan.zero) (CategoryTheory.Category …
  -/
  dsimp at w; simpa using w.symm
              /-
                🎉 no goals
              -/


/-- A pushout cocone on `f` and `g` is determined by morphisms `inl : Y ⟶ W` and `inr : Z ⟶ W` such
    that `f ≫ inl = g ↠ inr`. -/
@[simps]
def mk {W : C} (inl : Y ⟶ W) (inr : Z ⟶ W) (eq : f ≫ inl = g ≫ inr) : PushoutCocone f g where
  pt := W
  ι := { app := fun j => Option.casesOn j (f ≫ inl) fun j' => WalkingPair.casesOn j' inl inr
         naturality := by
          /-
            C : Type u
            inst✝ : CategoryTheory.Category.{v, u} C
            W✝ X Y Z : C
            f : Quiver.Hom X Y
            g : Quiver.Hom X Z
            W : C
            inl : Quiver.Hom Y W
            inr : Quiver.Hom Z W
            eq : Eq (CategoryTheory.CategoryStruct.comp f inl) (CategoryTheory.CategoryStr …
            ⊢ ∀ ⦃X_1 Y_1 : CategoryTheory.Limits.WalkingSpan⦄ (f_1 : Quiver.Hom X_1 Y_1),  …
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
                                                     /-
                                                       🎉 no goals
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
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
          rintro (⟨⟩|⟨⟨⟩⟩) (⟨⟩|⟨⟨⟩⟩) <;> intro f <;> cases f <;> dsimp <;> aesop }
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


@[simp]
theorem mk_ι_app_left {W : C} (inl : Y ⟶ W) (inr : Z ⟶ W) (eq : f ≫ inl = g ≫ inr) :
    (mk inl inr eq).ι.app WalkingSpan.left = inl := rfl


@[simp]
theorem mk_ι_app_right {W : C} (inl : Y ⟶ W) (inr : Z ⟶ W) (eq : f ≫ inl = g ≫ inr) :
    (mk inl inr eq).ι.app WalkingSpan.right = inr := rfl


@[simp]
theorem mk_ι_app_zero {W : C} (inl : Y ⟶ W) (inr : Z ⟶ W) (eq : f ≫ inl = g ≫ inr) :
    (mk inl inr eq).ι.app WalkingSpan.zero = f ≫ inl := rfl


@[simp]
theorem mk_inl {W : C} (inl : Y ⟶ W) (inr : Z ⟶ W) (eq : f ≫ inl = g ≫ inr) :
    (mk inl inr eq).inl = inl := rfl


@[simp]
theorem mk_inr {W : C} (inl : Y ⟶ W) (inr : Z ⟶ W) (eq : f ≫ inl = g ≫ inr) :
    (mk inl inr eq).inr = inr := rfl


@[reassoc]
theorem condition (t : PushoutCocone f g) : f ≫ inl t = g ≫ inr t :=
  (t.w fst).trans (t.w snd).symm


/-- To check whether a morphism is coequalized by the maps of a pushout cocone, it suffices to check
  it for `inl t` and `inr t` -/
theorem coequalizer_ext (t : PushoutCocone f g) {W : C} {k l : t.pt ⟶ W}
    (h₀ : inl t ≫ k = inl t ≫ l) (h₁ : inr t ≫ k = inr t ≫ l) :
    ∀ j : WalkingSpan, t.ι.app j ≫ k = t.ι.app j ≫ l
  | some WalkingPair.left => h₀
  | some WalkingPair.right => h₁
               /-
                 C : Type u
                 inst✝ : CategoryTheory.Category.{v, u} C
                 X Y Z : C
                 f : Quiver.Hom X Y
                 g : Quiver.Hom X Z
                 t : CategoryTheory.Limits.PushoutCocone f g
                 W : C
                 k l : Quiver.Hom t.pt W
                 h₀ : Eq (CategoryTheory.CategoryStruct.comp t.inl k) (CategoryTheory.CategoryS …
                 h₁ : Eq (CategoryTheory.CategoryStruct.comp t.inr k) (CategoryTheory.CategoryS …
                 ⊢ Eq (CategoryTheory.CategoryStruct.comp (t.ι.app Option.none) k) (CategoryThe …
               -/
  | none => by rw [← t.w fst, Category.assoc, Category.assoc, h₀]
               /-
                 🎉 no goals
               -/


/-- To construct an isomorphism of pushout cocones, it suffices to construct an isomorphism
of the cocone points and check it commutes with `inl` and `inr`. -/
def ext {s t : PushoutCocone f g} (i : s.pt ≅ t.pt) (w₁ : s.inl ≫ i.hom = t.inl := by aesop_cat)
    (w₂ : s.inr ≫ i.hom = t.inr := by aesop_cat) : s ≅ t :=
  WalkingSpan.ext i w₁ w₂


/-- The natural isomorphism between a pushout cocone and the corresponding pushout cocone
reconstructed using `PushoutCocone.mk`. -/
@[simps!]
def eta (t : PushoutCocone f g) : t ≅ mk t.inl t.inr t.condition :=
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    W X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom X Z
    t : CategoryTheory.Limits.PushoutCocone f g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp t.inl (CategoryTheory.Iso.refl t.pt). …
  -/
  /-
    🎉 no goals
  -/
  PushoutCocone.ext (Iso.refl _)
  /-
    🎉 no goals
  -/


/-- This is a slightly more convenient method to verify that a pushout cocone is a colimit cocone.
    It only asks for a proof of facts that carry any mathematical content -/
def isColimitAux (t : PushoutCocone f g) (desc : ∀ s : PushoutCocone f g, t.pt ⟶ s.pt)
    (fac_left : ∀ s : PushoutCocone f g, t.inl ≫ desc s = s.inl)
    (fac_right : ∀ s : PushoutCocone f g, t.inr ≫ desc s = s.inr)
    (uniq : ∀ (s : PushoutCocone f g) (m : t.pt ⟶ s.pt)
    (_ : ∀ j : WalkingSpan, t.ι.app j ≫ m = s.ι.app j), m = desc s) : IsColimit t :=
  { desc
    fac := fun s j =>
                           /-
                             C : Type u
                             inst✝ : CategoryTheory.Category.{v, u} C
                             W X Y Z : C
                             f : Quiver.Hom X Y
                             g : Quiver.Hom X Z
                             t : CategoryTheory.Limits.PushoutCocone f g
                             desc : (s : CategoryTheory.Limits.PushoutCocone f g) → Quiver.Hom t.pt s.pt
                             fac_left : ∀ (s : CategoryTheory.Limits.PushoutCocone f g), Eq (CategoryTheory …
                             fac_right : ∀ (s : CategoryTheory.Limits.PushoutCocone f g), Eq (CategoryTheor …
                             uniq : ∀ (s : CategoryTheory.Limits.PushoutCocone f g) (m : Quiver.Hom t.pt s. …
                             s : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.span f g)
                             j : CategoryTheory.Limits.WalkingSpan
                             ⊢ Eq (CategoryTheory.CategoryStruct.comp (t.ι.app Option.none) (desc s)) (s.ι. …
                           -/
      Option.casesOn j (by simp [← s.w fst, ← t.w fst, fac_left s]) fun j' =>
                           /-
                             🎉 no goals
                           -/
        WalkingPair.casesOn j' (fac_left s) (fac_right s)
    uniq := uniq }


/-- This is another convenient method to verify that a pushout cocone is a colimit cocone. It
    only asks for a proof of facts that carry any mathematical content, and allows access to the
    same `s` for all parts. -/
def isColimitAux' (t : PushoutCocone f g)
    (create :
      ∀ s : PushoutCocone f g,
        { l //
          t.inl ≫ l = s.inl ∧
            t.inr ≫ l = s.inr ∧ ∀ {m}, t.inl ≫ m = s.inl → t.inr ≫ m = s.inr → m = l }) :
    IsColimit t :=
  isColimitAux t (fun s => (create s).1) (fun s => (create s).2.1) (fun s => (create s).2.2.1)
    fun s _ w => (create s).2.2.2 (w WalkingCospan.left) (w WalkingCospan.right)



theorem IsColimit.hom_ext {t : PushoutCocone f g} (ht : IsColimit t) {W : C} {k l : t.pt ⟶ W}
    (h₀ : inl t ≫ k = inl t ≫ l) (h₁ : inr t ≫ k = inr t ≫ l) : k = l :=
  ht.hom_ext <| coequalizer_ext _ h₀ h₁

-- Porting note: `IsColimit.desc` and the two following simp lemmas were introduced to ease the port

/-- If `t` is a colimit pushout cocone over `f` and `g` and `h : Y ⟶ W` and `k : Z ⟶ W` are
    morphisms satisfying `f ≫ h = g ≫ k`, then we have a factorization `l : t.pt ⟶ W` such that
    `inl t ≫ l = h` and `inr t ≫ l = k`, see `IsColimit.inl_desc` and `IsColimit.inr_desc`-/
def IsColimit.desc {t : PushoutCocone f g} (ht : IsColimit t) {W : C} (h : Y ⟶ W) (k : Z ⟶ W)
    (w : f ≫ h = g ≫ k) : t.pt ⟶ W :=
  ht.desc (PushoutCocone.mk _ _ w)


@[reassoc (attr := simp)]
lemma IsColimit.inl_desc {t : PushoutCocone f g} (ht : IsColimit t) {W : C} (h : Y ⟶ W) (k : Z ⟶ W)
    (w : f ≫ h = g ≫ k) : inl t ≫ IsColimit.desc ht h k w = h :=
  ht.fac _ _


@[reassoc (attr := simp)]
lemma IsColimit.inr_desc {t : PushoutCocone f g} (ht : IsColimit t) {W : C} (h : Y ⟶ W) (k : Z ⟶ W)
    (w : f ≫ h = g ≫ k) : inr t ≫ IsColimit.desc ht h k w = k :=
  ht.fac _ _


/-- If `t` is a colimit pushout cocone over `f` and `g` and `h : Y ⟶ W` and `k : Z ⟶ W` are
    morphisms satisfying `f ≫ h = g ≫ k`, then we have a factorization `l : t.pt ⟶ W` such that
    `inl t ≫ l = h` and `inr t ≫ l = k`. -/
def IsColimit.desc' {t : PushoutCocone f g} (ht : IsColimit t) {W : C} (h : Y ⟶ W) (k : Z ⟶ W)
    (w : f ≫ h = g ≫ k) : { l : t.pt ⟶ W // inl t ≫ l = h ∧ inr t ≫ l = k } :=
                               /-
                                 C : Type u
                                 inst✝ : CategoryTheory.Category.{v, u} C
                                 W✝ X Y Z : C
                                 f : Quiver.Hom X Y
                                 g : Quiver.Hom X Z
                                 t : CategoryTheory.Limits.PushoutCocone f g
                                 ht : CategoryTheory.Limits.IsColimit t
                                 W : C
                                 h : Quiver.Hom Y W
                                 k : Quiver.Hom Z W
                                 w : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStruct …
                                 ⊢ And (Eq (CategoryTheory.CategoryStruct.comp t.inl (CategoryTheory.Limits.Pus …
                               -/
  ⟨IsColimit.desc ht h k w, by simp⟩
                               /-
                                 🎉 no goals
                               -/


/-- This is a more convenient formulation to show that a `PushoutCocone` constructed using
`PushoutCocone.mk` is a colimit cocone.
-/
def IsColimit.mk {W : C} {inl : Y ⟶ W} {inr : Z ⟶ W} (eq : f ≫ inl = g ≫ inr)
    (desc : ∀ s : PushoutCocone f g, W ⟶ s.pt)
    (fac_left : ∀ s : PushoutCocone f g, inl ≫ desc s = s.inl)
    (fac_right : ∀ s : PushoutCocone f g, inr ≫ desc s = s.inr)
    (uniq :
      ∀ (s : PushoutCocone f g) (m : W ⟶ s.pt) (_ : inl ≫ m = s.inl) (_ : inr ≫ m = s.inr),
        m = desc s) :
    IsColimit (mk inl inr eq) :=
  isColimitAux _ desc fac_left fac_right fun s m w =>
    uniq s m (w WalkingCospan.left) (w WalkingCospan.right)


/-- The pushout cocone reconstructed using `PushoutCocone.mk` from a pushout cocone that is a
colimit, is also a colimit. -/
def mkSelfIsColimit {t : PushoutCocone f g} (ht : IsColimit t) :
    IsColimit (mk t.inl t.inr t.condition) :=
  IsColimit.ofIsoColimit ht (eta t)


/-- The pushout cocone obtained by flipping `inl` and `inr`. -/
def flip : PushoutCocone g f := PushoutCocone.mk _ _ t.condition.symm


@[simp] lemma flip_inl : t.flip.inl = t.inr := rfl

@[simp] lemma flip_inr : t.flip.inr = t.inl := rfl


/-- Flipping a pushout cocone twice gives an isomorphic cocone. -/
                                                                        /-
                                                                          C : Type u
                                                                          inst✝ : CategoryTheory.Category.{v, u} C
                                                                          W X Y Z : C
                                                                          f : Quiver.Hom X Y
                                                                          g : Quiver.Hom X Z
                                                                          t : CategoryTheory.Limits.PushoutCocone f g
                                                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp t.flip.flip.inl (CategoryTheory.Iso.r …
                                                                        -/
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
def flipFlipIso : t.flip.flip ≅ t := PushoutCocone.ext (Iso.refl _) (by simp) (by simp)
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


/-- The flip of a pushout square is a pushout square. -/
def flipIsColimit (ht : IsColimit t) : IsColimit t.flip :=
                                               /-
                                                 C : Type u
                                                 inst✝ : CategoryTheory.Category.{v, u} C
                                                 W X Y Z : C
                                                 f : Quiver.Hom X Y
                                                 g : Quiver.Hom X Z
                                                 t : CategoryTheory.Limits.PushoutCocone f g
                                                 ht : CategoryTheory.Limits.IsColimit t
                                                 ⊢ ∀ (s : CategoryTheory.Limits.PushoutCocone g f), Eq (CategoryTheory.Category …
                                               -/
                                               /-
                                                 🎉 no goals
                                               -/
  IsColimit.mk _ (fun s => ht.desc s.flip) (by simp) (by simp) (fun s m h₁ h₂ => by
                                                         /-
                                                           🎉 no goals
                                                         -/
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      W X Y Z : C
      f : Quiver.Hom X Y
      g : Quiver.Hom X Z
      t : CategoryTheory.Limits.PushoutCocone f g
      ht : CategoryTheory.Limits.IsColimit t
      s : CategoryTheory.Limits.PushoutCocone g f
      m : Quiver.Hom t.pt s.pt
      h₁ : Eq (CategoryTheory.CategoryStruct.comp t.inr m) s.inl
      h₂ : Eq (CategoryTheory.CategoryStruct.comp t.inl m) s.inr
      ⊢ Eq m ((fun s => ht.desc s.flip) s)
    -/
                                   /-
                                     🎉 no goals
                                   -/
    apply IsColimit.hom_ext ht <;> simp [h₁, h₂])
                                   /-
                                     🎉 no goals
                                   -/


/-- A square is a pushout square if its flip is. -/
def isColimitOfFlip (ht : IsColimit t.flip) : IsColimit t :=
  IsColimit.ofIsoColimit (flipIsColimit ht) t.flipFlipIso


/-- This is a helper construction that can be useful when verifying that a category has all
    pushout. Given `F : WalkingSpan ⥤ C`, which is really the same as
    `span (F.map fst) (F.map snd)`, and a pushout cocone on `F.map fst` and `F.map snd`,
    we get a cocone on `F`.

    If you're thinking about using this, have a look at `hasPushouts_of_hasColimit_span`, which
    you may find to be an easier way of achieving your goal. -/
@[simps]
def Cocone.ofPushoutCocone {F : WalkingSpan ⥤ C} (t : PushoutCocone (F.map fst) (F.map snd)) :
    Cocone F where
  pt := t.pt
  ι := (diagramIsoSpan F).hom ≫ t.ι


/-- Given `F : WalkingSpan ⥤ C`, which is really the same as `span (F.map fst) (F.map snd)`,
    and a cocone on `F`, we get a pushout cocone on `F.map fst` and `F.map snd`. -/
@[simps]
def PushoutCocone.ofCocone {F : WalkingSpan ⥤ C} (t : Cocone F) :
    PushoutCocone (F.map fst) (F.map snd) where
  pt := t.pt
  ι := (diagramIsoSpan F).inv ≫ t.ι


/-- A diagram `WalkingSpan ⥤ C` is isomorphic to some `PushoutCocone.mk` after composing with
`diagramIsoSpan`. -/
@[simps!]
def PushoutCocone.isoMk {F : WalkingSpan ⥤ C} (t : Cocone F) :
    (Cocones.precompose (diagramIsoSpan.{v} _).inv).obj t ≅
      PushoutCocone.mk (t.ι.app WalkingSpan.left) (t.ι.app WalkingSpan.right)
        ((t.ι.naturality fst).trans (t.ι.naturality snd).symm) :=
  Cocones.ext (Iso.refl _) <| by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      W X Y Z : C
      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingSpan C
      t : CategoryTheory.Limits.Cocone F
      ⊢ ∀ (j : CategoryTheory.Limits.WalkingSpan), Eq (CategoryTheory.CategoryStruct …
    -/
    rintro (_ | (_ | _)) <;>
        /-
          case none
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          W X Y Z : C
          F : CategoryTheory.Functor CategoryTheory.Limits.WalkingSpan C
          t : CategoryTheory.Limits.Cocone F
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Limits.Cocones.prec …
        -/
        /-
          case none
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          W X Y Z : C
          F : CategoryTheory.Functor CategoryTheory.Limits.WalkingSpan C
          t : CategoryTheory.Limits.Cocone F
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
        /-
          case some.right
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          W X Y Z : C
          F : CategoryTheory.Functor CategoryTheory.Limits.WalkingSpan C
          t : CategoryTheory.Limits.Cocone F
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        simp
        /-
          🎉 no goals
        -/


