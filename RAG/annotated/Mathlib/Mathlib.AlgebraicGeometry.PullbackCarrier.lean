/-- A `Triplet` over `f : X ⟶ S` and `g : Y ⟶ S` is a triple of points `x : X`, `y : Y`,
`s : S` such that `f x = s = f y`. -/
structure Triplet {X Y S : Scheme.{u}} (f : X ⟶ S) (g : Y ⟶ S) where
  /-- The point of `X`. -/
  x : X
  /-- The point of `Y`. -/
  y : Y
  /-- The point of `S` below `x` and `y`. -/
  s : S
  hx : f.base x = s
  hy : g.base y = s


@[ext]
protected lemma ext {t₁ t₂ : Triplet f g} (ex : t₁.x = t₂.x) (ey : t₁.y = t₂.y) : t₁ = t₂ := by
  /-
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    t₁ t₂ : AlgebraicGeometry.Scheme.Pullback.Triplet f g
    ex : Eq t₁.x t₂.x
    ey : Eq t₁.y t₂.y
    ⊢ Eq t₁ t₂
  -/
  cases t₁; cases t₂; simp; aesop
                            /-
                              🎉 no goals
                            -/


/-- Make a triplet from `x : X` and `y : Y` such that `f x = g y`. -/
@[simps]
def mk' (x : X) (y : Y) (h : f.base x = g.base y) : Triplet f g where
  x := x
  y := y
  s := g.base y
  hx := h
  hy := rfl


/-- Given `x : X` and `y : Y` such that `f x = s = g y`, this is `κ(x) ⊗[κ(s)] κ(y)`. -/
def tensor (T : Triplet f g) : CommRingCat :=
  pushout ((S.residueFieldCongr T.hx).inv ≫ f.residueFieldMap T.x)
    ((S.residueFieldCongr T.hy).inv ≫ g.residueFieldMap T.y)


instance (T : Triplet f g) : Nontrivial T.tensor :=
  CommRingCat.nontrivial_of_isPushout_of_isField (Field.toIsField _)
    (IsPushout.of_hasPushout _ _)


/-- Given `x : X` and `y : Y` such that `f x = s = g y`, this is the
canonical map `κ(x) ⟶ κ(x) ⊗[κ(s)] κ(y)`. -/
def tensorInl (T : Triplet f g) : X.residueField T.x ⟶ T.tensor := pushout.inl _ _


/-- Given `x : X` and `y : Y` such that `f x = s = g y`, this is the
canonical map `κ(y) ⟶ κ(x) ⊗[κ(s)] κ(y)`. -/
def tensorInr (T : Triplet f g) : Y.residueField T.y ⟶ T.tensor := pushout.inr _ _


lemma Spec_map_tensor_isPullback (T : Triplet f g) : CategoryTheory.IsPullback
    (Spec.map T.tensorInl) (Spec.map T.tensorInr)
        (Spec.map ((S.residueFieldCongr T.hx).inv ≫ f.residueFieldMap T.x))
          (Spec.map ((S.residueFieldCongr T.hy).inv ≫ g.residueFieldMap T.y)) :=
  isPullback_Spec_map_pushout _ _


/-- Given propositionally equal triplets `T₁` and `T₂` over `f` and `g`, the corresponding
`T₁.tensor` and `T₂.tensor` are isomorphic. -/
def tensorCongr {T₁ T₂ : Triplet f g} (e : T₁ = T₂) :
    T₁.tensor ≅ T₂.tensor :=
              /-
                X Y S : AlgebraicGeometry.Scheme
                f : Quiver.Hom X S
                g : Quiver.Hom Y S
                T₁ T₂ : AlgebraicGeometry.Scheme.Pullback.Triplet f g
                e : Eq T₁ T₂
                ⊢ Eq T₁.tensor T₂.tensor
              -/
  eqToIso (by subst e; rfl)
                       /-
                         🎉 no goals
                       -/


@[simp]
lemma tensorCongr_refl {x : Triplet f g} :
    tensorCongr (refl x) = Iso.refl _ := rfl


@[simp]
lemma tensorCongr_symm {x y : Triplet f g} (e : x = y) :
    (tensorCongr e).symm = tensorCongr e.symm := rfl


@[simp]
lemma tensorCongr_inv {x y : Triplet f g} (e : x = y) :
    (tensorCongr e).inv = (tensorCongr e.symm).hom := rfl


@[simp]
lemma tensorCongr_trans {x y z : Triplet f g} (e : x = y) (e' : y = z) :
    tensorCongr e ≪≫ tensorCongr e' =
      tensorCongr (e.trans e') := by
  /-
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    x y z : AlgebraicGeometry.Scheme.Pullback.Triplet f g
    e : Eq x y
    e' : Eq y z
    ⊢ Eq ((AlgebraicGeometry.Scheme.Pullback.Triplet.tensorCongr e).trans (Algebra …
  -/
  subst e e'
  /-
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    x : AlgebraicGeometry.Scheme.Pullback.Triplet f g
    ⊢ Eq ((AlgebraicGeometry.Scheme.Pullback.Triplet.tensorCongr ⋯).trans (Algebra …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma tensorCongr_trans_hom {x y z : Triplet f g} (e : x = y) (e' : y = z) :
    (tensorCongr e).hom ≫ (tensorCongr e').hom =
      (tensorCongr (e.trans e')).hom := by
  /-
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    x y z : AlgebraicGeometry.Scheme.Pullback.Triplet f g
    e : Eq x y
    e' : Eq y z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Pullback.Tr …
  -/
  subst e e'
  /-
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    x : AlgebraicGeometry.Scheme.Pullback.Triplet f g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Pullback.Tr …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma Spec_map_tensorInl_fromSpecResidueField :
    (Spec.map T.tensorInl ≫ X.fromSpecResidueField T.x) ≫ f =
      (Spec.map T.tensorInr ≫ Y.fromSpecResidueField T.y) ≫ g := by
  simp only [residueFieldCongr_inv, Category.assoc, tensorInl, tensorInr,
    ← Hom.Spec_map_residueFieldMap_fromSpecResidueField]
  rw [← residueFieldCongr_fromSpecResidueField T.hx.symm,
    ← residueFieldCongr_fromSpecResidueField T.hy.symm]
  /-
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    T : AlgebraicGeometry.Scheme.Pullback.Triplet f g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Category …
  -/
  simp only [← Category.assoc, ← Spec.map_comp, pushout.condition]
  /-
    🎉 no goals
  -/


/-- Given `x : X`, `y : Y` and `s : S` such that `f x = s = g y`,
this is `Spec (κ(x) ⊗[κ(s)] κ(y)) ⟶ X ×ₛ Y`. -/
def SpecTensorTo : Spec T.tensor ⟶ pullback f g :=
  pullback.lift (Spec.map T.tensorInl ≫ X.fromSpecResidueField T.x)
    (Spec.map T.tensorInr ≫ Y.fromSpecResidueField T.y)
    (Spec_map_tensorInl_fromSpecResidueField _)


@[simp]
lemma specTensorTo_base_fst (p : Spec T.tensor) :
    (pullback.fst f g).base (T.SpecTensorTo.base p) = T.x := by
  /-
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    T : AlgebraicGeometry.Scheme.Pullback.Triplet f g
    p : ↑↑(AlgebraicGeometry.Spec T.tensor).toPresheafedSpace
    ⊢ Eq ((CategoryTheory.Limits.pullback.fst f g).base (T.SpecTensorTo.base p)) T.x
  -/
  simp only [SpecTensorTo, residueFieldCongr_inv]
  /-
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    T : AlgebraicGeometry.Scheme.Pullback.Triplet f g
    p : ↑↑(AlgebraicGeometry.Spec T.tensor).toPresheafedSpace
    ⊢ Eq ((CategoryTheory.Limits.pullback.fst f g).base ((CategoryTheory.Limits.pu …
  -/
  rw [← Scheme.comp_base_apply]
  /-
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    T : AlgebraicGeometry.Scheme.Pullback.Triplet f g
    p : ↑↑(AlgebraicGeometry.Spec T.tensor).toPresheafedSpace
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.lift …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma specTensorTo_base_snd (p : Spec T.tensor) :
    (pullback.snd f g).base (T.SpecTensorTo.base p) = T.y := by
  /-
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    T : AlgebraicGeometry.Scheme.Pullback.Triplet f g
    p : ↑↑(AlgebraicGeometry.Spec T.tensor).toPresheafedSpace
    ⊢ Eq ((CategoryTheory.Limits.pullback.snd f g).base (T.SpecTensorTo.base p)) T.y
  -/
  simp only [SpecTensorTo, residueFieldCongr_inv]
  /-
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    T : AlgebraicGeometry.Scheme.Pullback.Triplet f g
    p : ↑↑(AlgebraicGeometry.Spec T.tensor).toPresheafedSpace
    ⊢ Eq ((CategoryTheory.Limits.pullback.snd f g).base ((CategoryTheory.Limits.pu …
  -/
  rw [← Scheme.comp_base_apply]
  /-
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    T : AlgebraicGeometry.Scheme.Pullback.Triplet f g
    p : ↑↑(AlgebraicGeometry.Spec T.tensor).toPresheafedSpace
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.lift …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma specTensorTo_fst :
    T.SpecTensorTo ≫ pullback.fst f g = Spec.map T.tensorInl ≫ X.fromSpecResidueField T.x :=
  pullback.lift_fst _ _ _


@[reassoc (attr := simp)]
lemma specTensorTo_snd :
    T.SpecTensorTo ≫ pullback.snd f g = Spec.map T.tensorInr ≫ Y.fromSpecResidueField T.y :=
  pullback.lift_snd _ _ _


/-- Given `t : X ×[S] Y`, it maps to `X` and `Y` with same image in `S`, yielding a
`Triplet f g`. -/
@[simps]
def ofPoint (t : ↑(pullback f g)) : Triplet f g :=
  ⟨(pullback.fst f g).base t, (pullback.snd f g).base t, _, rfl,
    congr((Scheme.Hom.toLRSHom $(pullback.condition (f := f) (g := g))).base t).symm⟩


@[simp]
lemma ofPoint_SpecTensorTo (T : Triplet f g) (p : Spec T.tensor) :
    ofPoint (T.SpecTensorTo.base p) = T := by
  /-
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    T : AlgebraicGeometry.Scheme.Pullback.Triplet f g
    p : ↑↑(AlgebraicGeometry.Spec T.tensor).toPresheafedSpace
    ⊢ Eq (AlgebraicGeometry.Scheme.Pullback.Triplet.ofPoint (T.SpecTensorTo.base p …
  -/
          /-
            🎉 no goals
          -/
  ext <;> simp
          /-
            🎉 no goals
          -/


lemma residueFieldCongr_inv_residueFieldMap_ofPoint (t : ↑(pullback f g)) :
    ((S.residueFieldCongr (Triplet.ofPoint t).hx).inv ≫ f.residueFieldMap (Triplet.ofPoint t).x) ≫
      (pullback.fst f g).residueFieldMap t = ((S.residueFieldCongr (Triplet.ofPoint t).hy).inv ≫
          g.residueFieldMap (Triplet.ofPoint t).y) ≫ (pullback.snd f g).residueFieldMap t := by
  /-
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    t : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp [← residueFieldMap_comp, Scheme.Hom.residueFieldMap_congr pullback.condition]
  /-
    🎉 no goals
  -/


/-- Given `t : X ×[S] Y` with projections to `X`, `Y` and `S` denoted by `x`, `y` and `s`
respectively, this is the canonical map `κ(x) ⊗[κ(s)] κ(y) ⟶ κ(t)`. -/
def ofPointTensor (t : ↑(pullback f g)) :
    (Triplet.ofPoint t).tensor ⟶ (pullback f g).residueField t :=
  pushout.desc
    ((pullback.fst f g).residueFieldMap t)
    ((pullback.snd f g).residueFieldMap t)
    (residueFieldCongr_inv_residueFieldMap_ofPoint t)


@[reassoc]
lemma ofPointTensor_SpecTensorTo (t : ↑(pullback f g)) :
    Spec.map (ofPointTensor t) ≫ (Triplet.ofPoint t).SpecTensorTo =
      (pullback f g).fromSpecResidueField t := by
  /-
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    t : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Algebrai …
  -/
  apply pullback.hom_ext
    /-
      case h₀
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      t : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
  · rw [← Scheme.Hom.Spec_map_residueFieldMap_fromSpecResidueField]
    /-
      case h₀
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      t : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp only [Category.assoc, Triplet.specTensorTo_fst, Triplet.ofPoint_x]
    /-
      case h₀
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      t : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Algebrai …
    -/
    rw [← pushout.inl_desc _ _ (residueFieldCongr_inv_residueFieldMap_ofPoint t), Spec.map_comp]
    /-
      case h₀
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      t : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Algebrai …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case h₁
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      t : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
  · rw [← Scheme.Hom.Spec_map_residueFieldMap_fromSpecResidueField]
    /-
      case h₁
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      t : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp only [Category.assoc, Triplet.specTensorTo_snd, Triplet.ofPoint_y]
    /-
      case h₁
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      t : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Algebrai …
    -/
    rw [← pushout.inr_desc _ _ (residueFieldCongr_inv_residueFieldMap_ofPoint t), Spec.map_comp]
    /-
      case h₁
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      t : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Algebrai …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- If `t` is a point in `X ×[S] Y` above `(x, y, s)`, then this is the image of the unique
point of `Spec κ(s)` in `Spec κ(x) ⊗[κ(s)] κ(y)`. -/
def SpecOfPoint (t : ↑(pullback f g)) : Spec (Triplet.ofPoint t).tensor :=
    (Spec.map (ofPointTensor t)).base (⊥ : PrimeSpectrum _)


@[simp]
lemma SpecTensorTo_SpecOfPoint (t : ↑(pullback f g)) :
    (Triplet.ofPoint t).SpecTensorTo.base (SpecOfPoint t) = t := by
  /-
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    t : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace
    ⊢ Eq ((AlgebraicGeometry.Scheme.Pullback.Triplet.ofPoint t).SpecTensorTo.base  …
  -/
  simp [SpecOfPoint, ← Scheme.comp_base_apply, ofPointTensor_SpecTensorTo]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma tensorCongr_SpecTensorTo {T T' : Triplet f g} (h : T = T') :
    Spec.map (Triplet.tensorCongr h).hom ≫ T.SpecTensorTo = T'.SpecTensorTo := by
  /-
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    T T' : AlgebraicGeometry.Scheme.Pullback.Triplet f g
    h : Eq T T'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Algebrai …
  -/
  subst h
  /-
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    T : AlgebraicGeometry.Scheme.Pullback.Triplet f g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Algebrai …
  -/
  simp only [Triplet.tensorCongr_refl, Iso.refl_hom, Spec.map_id, Category.id_comp]
  /-
    🎉 no goals
  -/


lemma Triplet.Spec_ofPointTensor_SpecTensorTo (T : Triplet f g) (p : Spec T.tensor) :
    Spec.map (Hom.residueFieldMap T.SpecTensorTo p) ≫
      Spec.map (ofPointTensor (T.SpecTensorTo.base p)) ≫
      Spec.map (tensorCongr (T.ofPoint_SpecTensorTo p).symm).hom =
    (Spec T.tensor).fromSpecResidueField p := by
  /-
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    T : AlgebraicGeometry.Scheme.Pullback.Triplet f g
    p : ↑↑(AlgebraicGeometry.Spec T.tensor).toPresheafedSpace
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Algebrai …
  -/
  apply T.Spec_map_tensor_isPullback.hom_ext
    /-
      case h₀
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      T : AlgebraicGeometry.Scheme.Pullback.Triplet f g
      p : ↑↑(AlgebraicGeometry.Spec T.tensor).toPresheafedSpace
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
  · rw [← cancel_mono <| X.fromSpecResidueField T.x]
    /-
      case h₀
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      T : AlgebraicGeometry.Scheme.Pullback.Triplet f g
      p : ↑↑(AlgebraicGeometry.Spec T.tensor).toPresheafedSpace
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp_rw [Category.assoc, ← T.specTensorTo_fst, tensorCongr_SpecTensorTo_assoc]
    /-
      case h₀
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      T : AlgebraicGeometry.Scheme.Pullback.Triplet f g
      p : ↑↑(AlgebraicGeometry.Spec T.tensor).toPresheafedSpace
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Algebrai …
    -/
    rw [← Hom.Spec_map_residueFieldMap_fromSpecResidueField_assoc, ofPointTensor_SpecTensorTo_assoc]
    /-
      🎉 no goals
    -/
    /-
      case h₁
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      T : AlgebraicGeometry.Scheme.Pullback.Triplet f g
      p : ↑↑(AlgebraicGeometry.Spec T.tensor).toPresheafedSpace
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
  · rw [← cancel_mono <| Y.fromSpecResidueField T.y]
    /-
      case h₁
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      T : AlgebraicGeometry.Scheme.Pullback.Triplet f g
      p : ↑↑(AlgebraicGeometry.Spec T.tensor).toPresheafedSpace
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp_rw [Category.assoc, ← T.specTensorTo_snd, tensorCongr_SpecTensorTo_assoc]
    /-
      case h₁
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      T : AlgebraicGeometry.Scheme.Pullback.Triplet f g
      p : ↑↑(AlgebraicGeometry.Spec T.tensor).toPresheafedSpace
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Algebrai …
    -/
    rw [← Hom.Spec_map_residueFieldMap_fromSpecResidueField_assoc, ofPointTensor_SpecTensorTo_assoc]
    /-
      🎉 no goals
    -/


/-- A helper lemma to work with `AlgebraicGeometry.Scheme.Pullback.carrierEquiv`. -/
lemma carrierEquiv_eq_iff {T₁ T₂ : Σ T : Triplet f g, Spec T.tensor} :
    T₁ = T₂ ↔ ∃ e : T₁.1 = T₂.1, (Spec.map (Triplet.tensorCongr e).inv).base T₁.2 = T₂.2 := by
  /-
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    T₁ T₂ : Sigma fun T => ↑↑(AlgebraicGeometry.Spec T.tensor).toPresheafedSpace
    ⊢ Iff (Eq T₁ T₂) (Exists fun e => Eq ((AlgebraicGeometry.Spec.map (AlgebraicGe …
  -/
  constructor
    /-
      case mp
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      T₁ T₂ : Sigma fun T => ↑↑(AlgebraicGeometry.Spec T.tensor).toPresheafedSpace
      ⊢ Eq T₁ T₂ → Exists fun e => Eq ((AlgebraicGeometry.Spec.map (AlgebraicGeometr …
    -/
  · rintro rfl
    /-
      case mp
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      T₁ : Sigma fun T => ↑↑(AlgebraicGeometry.Spec T.tensor).toPresheafedSpace
      ⊢ Exists fun e => Eq ((AlgebraicGeometry.Spec.map (AlgebraicGeometry.Scheme.Pu …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case mpr
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      T₁ T₂ : Sigma fun T => ↑↑(AlgebraicGeometry.Spec T.tensor).toPresheafedSpace
      ⊢ (Exists fun e => Eq ((AlgebraicGeometry.Spec.map (AlgebraicGeometry.Scheme.P …
    -/
  · obtain ⟨T, _⟩ := T₁
    /-
      case mpr.mk
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      T₂ : Sigma fun T => ↑↑(AlgebraicGeometry.Spec T.tensor).toPresheafedSpace
      T : AlgebraicGeometry.Scheme.Pullback.Triplet f g
      snd✝ : ↑↑(AlgebraicGeometry.Spec T.tensor).toPresheafedSpace
      ⊢ (Exists fun e => Eq ((AlgebraicGeometry.Spec.map (AlgebraicGeometry.Scheme.P …
    -/
    obtain ⟨T', _⟩ := T₂
    /-
      case mpr.mk.mk
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      T : AlgebraicGeometry.Scheme.Pullback.Triplet f g
      snd✝¹ : ↑↑(AlgebraicGeometry.Spec T.tensor).toPresheafedSpace
      T' : AlgebraicGeometry.Scheme.Pullback.Triplet f g
      snd✝ : ↑↑(AlgebraicGeometry.Spec T'.tensor).toPresheafedSpace
      ⊢ (Exists fun e => Eq ((AlgebraicGeometry.Spec.map (AlgebraicGeometry.Scheme.P …
    -/
    rintro ⟨rfl : T = T', e⟩
    /-
      case mpr.mk.mk.intro
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      T : AlgebraicGeometry.Scheme.Pullback.Triplet f g
      snd✝¹ snd✝ : ↑↑(AlgebraicGeometry.Spec T.tensor).toPresheafedSpace
      e : Eq ((AlgebraicGeometry.Spec.map (AlgebraicGeometry.Scheme.Pullback.Triplet …
      ⊢ Eq ⟨T, snd✝¹⟩ ⟨T, snd✝⟩
    -/
    simpa [e]
    /-
      🎉 no goals
    -/


/--
The points of the underlying topological space of `X ×[S] Y` bijectively correspond to
pairs of triples `x : X`, `y : Y`, `s : S` with `f x = s = f y` and prime ideals of
`κ(x) ⊗[κ(s)] κ(y)`.
-/
def carrierEquiv : ↑(pullback f g) ≃ Σ T : Triplet f g, Spec T.tensor where
  toFun t := ⟨.ofPoint t, SpecOfPoint t⟩
  invFun T := T.1.SpecTensorTo.base T.2
  left_inv := SpecTensorTo_SpecOfPoint
  right_inv := by
    /-
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      ⊢ Function.RightInverse (fun T => T.fst.SpecTensorTo.base T.snd) fun t => ⟨Alg …
    -/
    intro ⟨T, p⟩
    /-
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      T : AlgebraicGeometry.Scheme.Pullback.Triplet f g
      p : ↑↑(AlgebraicGeometry.Spec T.tensor).toPresheafedSpace
      ⊢ Eq ((fun t => ⟨AlgebraicGeometry.Scheme.Pullback.Triplet.ofPoint t, Algebrai …
    -/
    apply carrierEquiv_eq_iff.mpr
    /-
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      T : AlgebraicGeometry.Scheme.Pullback.Triplet f g
      p : ↑↑(AlgebraicGeometry.Spec T.tensor).toPresheafedSpace
      ⊢ Exists fun e => Eq ((AlgebraicGeometry.Spec.map (AlgebraicGeometry.Scheme.Pu …
    -/
    use T.ofPoint_SpecTensorTo p
    have : (Spec.map (Hom.residueFieldMap T.SpecTensorTo p)).base (⊥ : PrimeSpectrum _) =
        (⊥ : PrimeSpectrum _) :=
      (PrimeSpectrum.instUnique).uniq _
    simp only [SpecOfPoint, Triplet.tensorCongr_inv, ← this, ← Scheme.comp_base_apply,
      ← Scheme.comp_base_apply]
    /-
      case h
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      T : AlgebraicGeometry.Scheme.Pullback.Triplet f g
      p : ↑↑(AlgebraicGeometry.Spec T.tensor).toPresheafedSpace
      this : Eq ((AlgebraicGeometry.Spec.map (AlgebraicGeometry.Scheme.Hom.residueFi …
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp  …
    -/
    simp [Triplet.Spec_ofPointTensor_SpecTensorTo]
    /-
      🎉 no goals
    -/


@[simp]
lemma carrierEquiv_symm_fst (T : Triplet f g) (p : Spec T.tensor) :
    (pullback.fst f g).base (carrierEquiv.symm ⟨T, p⟩) = T.x := by
  /-
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    T : AlgebraicGeometry.Scheme.Pullback.Triplet f g
    p : ↑↑(AlgebraicGeometry.Spec T.tensor).toPresheafedSpace
    ⊢ Eq ((CategoryTheory.Limits.pullback.fst f g).base (AlgebraicGeometry.Scheme. …
  -/
  simp [carrierEquiv]
  /-
    🎉 no goals
  -/


@[simp]
lemma carrierEquiv_symm_snd (T : Triplet f g) (p : Spec T.tensor) :
    (pullback.snd f g).base (carrierEquiv.symm ⟨T, p⟩) = T.y := by
  /-
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    T : AlgebraicGeometry.Scheme.Pullback.Triplet f g
    p : ↑↑(AlgebraicGeometry.Spec T.tensor).toPresheafedSpace
    ⊢ Eq ((CategoryTheory.Limits.pullback.snd f g).base (AlgebraicGeometry.Scheme. …
  -/
  simp [carrierEquiv]
  /-
    🎉 no goals
  -/


/-- Given a triple `(x, y, s)` with `f x = s = f y` there exists `t : X ×[S] Y` above
`x` and `ỳ`. For the unpacked version without `Triplet`, see
`AlgebraicGeometry.Scheme.Pullback.exists_preimage`. -/
lemma Triplet.exists_preimage (T : Triplet f g) :
    ∃ t : ↑(pullback f g),
    (pullback.fst f g).base t = T.x ∧ (pullback.snd f g).base t = T.y :=
                                                          /-
                                                            X Y S : AlgebraicGeometry.Scheme
                                                            f : Quiver.Hom X S
                                                            g : Quiver.Hom Y S
                                                            T : AlgebraicGeometry.Scheme.Pullback.Triplet f g
                                                            ⊢ And (Eq ((CategoryTheory.Limits.pullback.fst f g).base (AlgebraicGeometry.Sc …
                                                          -/
  ⟨carrierEquiv.symm ⟨T, Nonempty.some inferInstance⟩, by simp⟩
                                                          /-
                                                            🎉 no goals
                                                          -/


/--
If `f : X ⟶ S` and `g : Y ⟶ S` are morphisms of schemes and `x : X` and `y : Y` are points such
that `f x = g y`, then there exists `z : X ×[S] Y` lying above `x` and `y`.

In other words, the map from the underlying topological space of `X ×[S] Y` to the fiber product
of the underlying topological spaces of `X` and `Y` over `S` is surjective.
-/
lemma exists_preimage_pullback (x : X) (y : Y) (h : f.base x = g.base y) :
    ∃ z : ↑(pullback f g),
    (pullback.fst f g).base z = x ∧ (pullback.snd f g).base z = y :=
  (Pullback.Triplet.mk' x y h).exists_preimage


lemma range_fst : Set.range (pullback.fst f g).base = f.base ⁻¹' Set.range g.base := by
  /-
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    ⊢ Eq (Set.range ⇑(CategoryTheory.Limits.pullback.fst f g).base) (Set.preimage  …
  -/
  ext x
  /-
    case h
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    x : ↑↑X.toPresheafedSpace
    ⊢ Iff (Membership.mem (Set.range ⇑(CategoryTheory.Limits.pullback.fst f g).bas …
  -/
  refine ⟨?_, fun ⟨y, hy⟩ ↦ ?_⟩
    /-
      case h.refine_1
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      x : ↑↑X.toPresheafedSpace
      ⊢ Membership.mem (Set.range ⇑(CategoryTheory.Limits.pullback.fst f g).base) x  …
    -/
  · rintro ⟨a, rfl⟩
    /-
      case h.refine_1.intro
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      a : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace
      ⊢ Membership.mem (Set.preimage (⇑f.base) (Set.range ⇑g.base)) ((CategoryTheory …
    -/
    simp only [Set.mem_preimage, Set.mem_range, ← Scheme.comp_base_apply, pullback.condition]
    /-
      case h.refine_1.intro
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      a : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace
      ⊢ Exists fun y => Eq (g.base y) ((CategoryTheory.CategoryStruct.comp (Category …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      x : ↑↑X.toPresheafedSpace
      x✝ : Membership.mem (Set.preimage (⇑f.base) (Set.range ⇑g.base)) x
      y : ↑↑Y.toPresheafedSpace
      hy : Eq (g.base y) (f.base x)
      ⊢ Membership.mem (Set.range ⇑(CategoryTheory.Limits.pullback.fst f g).base) x
    -/
  · obtain ⟨a, ha⟩ := Triplet.exists_preimage (Triplet.mk' x y hy.symm)
    /-
      case h.refine_2.intro
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      x : ↑↑X.toPresheafedSpace
      x✝ : Membership.mem (Set.preimage (⇑f.base) (Set.range ⇑g.base)) x
      y : ↑↑Y.toPresheafedSpace
      hy : Eq (g.base y) (f.base x)
      a : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace
      ha : And (Eq ((CategoryTheory.Limits.pullback.fst f g).base a) (AlgebraicGeome …
      ⊢ Membership.mem (Set.range ⇑(CategoryTheory.Limits.pullback.fst f g).base) x
    -/
    use a, ha.left
    /-
      🎉 no goals
    -/


lemma range_snd : Set.range (pullback.snd f g).base = g.base ⁻¹' Set.range f.base := by
  /-
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    ⊢ Eq (Set.range ⇑(CategoryTheory.Limits.pullback.snd f g).base) (Set.preimage  …
  -/
  ext x
  /-
    case h
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    x : ↑↑Y.toPresheafedSpace
    ⊢ Iff (Membership.mem (Set.range ⇑(CategoryTheory.Limits.pullback.snd f g).bas …
  -/
  refine ⟨?_, fun ⟨y, hy⟩ ↦ ?_⟩
    /-
      case h.refine_1
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      x : ↑↑Y.toPresheafedSpace
      ⊢ Membership.mem (Set.range ⇑(CategoryTheory.Limits.pullback.snd f g).base) x  …
    -/
  · rintro ⟨a, rfl⟩
    /-
      case h.refine_1.intro
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      a : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace
      ⊢ Membership.mem (Set.preimage (⇑g.base) (Set.range ⇑f.base)) ((CategoryTheory …
    -/
    simp only [Set.mem_preimage, Set.mem_range, ← Scheme.comp_base_apply, ← pullback.condition]
    /-
      case h.refine_1.intro
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      a : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace
      ⊢ Exists fun y => Eq (f.base y) ((CategoryTheory.CategoryStruct.comp (Category …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      x : ↑↑Y.toPresheafedSpace
      x✝ : Membership.mem (Set.preimage (⇑g.base) (Set.range ⇑f.base)) x
      y : ↑↑X.toPresheafedSpace
      hy : Eq (f.base y) (g.base x)
      ⊢ Membership.mem (Set.range ⇑(CategoryTheory.Limits.pullback.snd f g).base) x
    -/
  · obtain ⟨a, ha⟩ := Triplet.exists_preimage (Triplet.mk' y x hy)
    /-
      case h.refine_2.intro
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      x : ↑↑Y.toPresheafedSpace
      x✝ : Membership.mem (Set.preimage (⇑g.base) (Set.range ⇑f.base)) x
      y : ↑↑X.toPresheafedSpace
      hy : Eq (f.base y) (g.base x)
      a : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace
      ha : And (Eq ((CategoryTheory.Limits.pullback.fst f g).base a) (AlgebraicGeome …
      ⊢ Membership.mem (Set.range ⇑(CategoryTheory.Limits.pullback.snd f g).base) x
    -/
    use a, ha.right
    /-
      🎉 no goals
    -/


lemma range_fst_comp :
    Set.range (pullback.fst f g ≫ f).base = Set.range f.base ∩ Set.range g.base := by
  /-
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    ⊢ Eq (Set.range ⇑(CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pu …
  -/
  simp [Set.range_comp, range_fst, Set.image_preimage_eq_range_inter]
  /-
    🎉 no goals
  -/


lemma range_snd_comp :
    Set.range (pullback.snd f g ≫ g).base = Set.range f.base ∩ Set.range g.base := by
  /-
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    ⊢ Eq (Set.range ⇑(CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pu …
  -/
  rw [← pullback.condition, range_fst_comp]
  /-
    🎉 no goals
  -/


lemma range_map {X' Y' S' : Scheme.{u}} (f' : X' ⟶ S') (g' : Y' ⟶ S') (i₁ : X ⟶ X')
    (i₂ : Y ⟶ Y') (i₃ : S ⟶ S') (e₁ : f ≫ i₃ = i₁ ≫ f')
    (e₂ : g ≫ i₃ = i₂ ≫ g') [Mono i₃] :
    Set.range (pullback.map f g f' g' i₁ i₂ i₃ e₁ e₂).base =
      (pullback.fst f' g').base ⁻¹' Set.range i₁.base ∩
        (pullback.snd f' g').base ⁻¹' Set.range i₂.base := by
  /-
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    X' Y' S' : AlgebraicGeometry.Scheme
    f' : Quiver.Hom X' S'
    g' : Quiver.Hom Y' S'
    i₁ : Quiver.Hom X X'
    i₂ : Quiver.Hom Y Y'
    i₃ : Quiver.Hom S S'
    e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₃) (CategoryTheory.CategoryStru …
    e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
    inst✝ : CategoryTheory.Mono i₃
    ⊢ Eq (Set.range ⇑(CategoryTheory.Limits.pullback.map f g f' g' i₁ i₂ i₃ e₁ e₂) …
  -/
  ext z
  /-
    case h
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    X' Y' S' : AlgebraicGeometry.Scheme
    f' : Quiver.Hom X' S'
    g' : Quiver.Hom Y' S'
    i₁ : Quiver.Hom X X'
    i₂ : Quiver.Hom Y Y'
    i₃ : Quiver.Hom S S'
    e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₃) (CategoryTheory.CategoryStru …
    e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
    inst✝ : CategoryTheory.Mono i₃
    z : ↑↑(CategoryTheory.Limits.pullback f' g').toPresheafedSpace
    ⊢ Iff (Membership.mem (Set.range ⇑(CategoryTheory.Limits.pullback.map f g f' g …
  -/
  constructor
    /-
      case h.mp
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      X' Y' S' : AlgebraicGeometry.Scheme
      f' : Quiver.Hom X' S'
      g' : Quiver.Hom Y' S'
      i₁ : Quiver.Hom X X'
      i₂ : Quiver.Hom Y Y'
      i₃ : Quiver.Hom S S'
      e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₃) (CategoryTheory.CategoryStru …
      e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
      inst✝ : CategoryTheory.Mono i₃
      z : ↑↑(CategoryTheory.Limits.pullback f' g').toPresheafedSpace
      ⊢ Membership.mem (Set.range ⇑(CategoryTheory.Limits.pullback.map f g f' g' i₁  …
    -/
  · rintro ⟨t, rfl⟩
    /-
      case h.mp.intro
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      X' Y' S' : AlgebraicGeometry.Scheme
      f' : Quiver.Hom X' S'
      g' : Quiver.Hom Y' S'
      i₁ : Quiver.Hom X X'
      i₂ : Quiver.Hom Y Y'
      i₃ : Quiver.Hom S S'
      e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₃) (CategoryTheory.CategoryStru …
      e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
      inst✝ : CategoryTheory.Mono i₃
      t : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace
      ⊢ Membership.mem (Inter.inter (Set.preimage (⇑(CategoryTheory.Limits.pullback. …
    -/
    constructor
      /-
        case h.mp.intro.left
        X Y S : AlgebraicGeometry.Scheme
        f : Quiver.Hom X S
        g : Quiver.Hom Y S
        X' Y' S' : AlgebraicGeometry.Scheme
        f' : Quiver.Hom X' S'
        g' : Quiver.Hom Y' S'
        i₁ : Quiver.Hom X X'
        i₂ : Quiver.Hom Y Y'
        i₃ : Quiver.Hom S S'
        e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₃) (CategoryTheory.CategoryStru …
        e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
        inst✝ : CategoryTheory.Mono i₃
        t : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace
        ⊢ Membership.mem (Set.preimage (⇑(CategoryTheory.Limits.pullback.fst f' g').ba …
      -/
    · use (pullback.fst f g).base t
      /-
        case h
        X Y S : AlgebraicGeometry.Scheme
        f : Quiver.Hom X S
        g : Quiver.Hom Y S
        X' Y' S' : AlgebraicGeometry.Scheme
        f' : Quiver.Hom X' S'
        g' : Quiver.Hom Y' S'
        i₁ : Quiver.Hom X X'
        i₂ : Quiver.Hom Y Y'
        i₃ : Quiver.Hom S S'
        e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₃) (CategoryTheory.CategoryStru …
        e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
        inst✝ : CategoryTheory.Mono i₃
        t : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace
        ⊢ Eq (i₁.base ((CategoryTheory.Limits.pullback.fst f g).base t)) ((CategoryThe …
      -/
      rw [← Scheme.comp_base_apply, ← Scheme.comp_base_apply]
      /-
        case h
        X Y S : AlgebraicGeometry.Scheme
        f : Quiver.Hom X S
        g : Quiver.Hom Y S
        X' Y' S' : AlgebraicGeometry.Scheme
        f' : Quiver.Hom X' S'
        g' : Quiver.Hom Y' S'
        i₁ : Quiver.Hom X X'
        i₂ : Quiver.Hom Y Y'
        i₃ : Quiver.Hom S S'
        e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₃) (CategoryTheory.CategoryStru …
        e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
        inst✝ : CategoryTheory.Mono i₃
        t : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fst  …
      -/
      simp
      /-
        🎉 no goals
      -/
      /-
        case h.mp.intro.right
        X Y S : AlgebraicGeometry.Scheme
        f : Quiver.Hom X S
        g : Quiver.Hom Y S
        X' Y' S' : AlgebraicGeometry.Scheme
        f' : Quiver.Hom X' S'
        g' : Quiver.Hom Y' S'
        i₁ : Quiver.Hom X X'
        i₂ : Quiver.Hom Y Y'
        i₃ : Quiver.Hom S S'
        e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₃) (CategoryTheory.CategoryStru …
        e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
        inst✝ : CategoryTheory.Mono i₃
        t : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace
        ⊢ Membership.mem (Set.preimage (⇑(CategoryTheory.Limits.pullback.snd f' g').ba …
      -/
    · use (pullback.snd f g).base t
      /-
        case h
        X Y S : AlgebraicGeometry.Scheme
        f : Quiver.Hom X S
        g : Quiver.Hom Y S
        X' Y' S' : AlgebraicGeometry.Scheme
        f' : Quiver.Hom X' S'
        g' : Quiver.Hom Y' S'
        i₁ : Quiver.Hom X X'
        i₂ : Quiver.Hom Y Y'
        i₃ : Quiver.Hom S S'
        e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₃) (CategoryTheory.CategoryStru …
        e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
        inst✝ : CategoryTheory.Mono i₃
        t : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace
        ⊢ Eq (i₂.base ((CategoryTheory.Limits.pullback.snd f g).base t)) ((CategoryThe …
      -/
      rw [← Scheme.comp_base_apply, ← Scheme.comp_base_apply]
      /-
        case h
        X Y S : AlgebraicGeometry.Scheme
        f : Quiver.Hom X S
        g : Quiver.Hom Y S
        X' Y' S' : AlgebraicGeometry.Scheme
        f' : Quiver.Hom X' S'
        g' : Quiver.Hom Y' S'
        i₁ : Quiver.Hom X X'
        i₂ : Quiver.Hom Y Y'
        i₃ : Quiver.Hom S S'
        e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₃) (CategoryTheory.CategoryStru …
        e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
        inst✝ : CategoryTheory.Mono i₃
        t : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.snd  …
      -/
      simp
      /-
        🎉 no goals
      -/
    /-
      case h.mpr
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      X' Y' S' : AlgebraicGeometry.Scheme
      f' : Quiver.Hom X' S'
      g' : Quiver.Hom Y' S'
      i₁ : Quiver.Hom X X'
      i₂ : Quiver.Hom Y Y'
      i₃ : Quiver.Hom S S'
      e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₃) (CategoryTheory.CategoryStru …
      e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
      inst✝ : CategoryTheory.Mono i₃
      z : ↑↑(CategoryTheory.Limits.pullback f' g').toPresheafedSpace
      ⊢ Membership.mem (Inter.inter (Set.preimage (⇑(CategoryTheory.Limits.pullback. …
    -/
  · intro ⟨⟨x, hx⟩, ⟨y, hy⟩⟩
    /-
      case h.mpr
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      X' Y' S' : AlgebraicGeometry.Scheme
      f' : Quiver.Hom X' S'
      g' : Quiver.Hom Y' S'
      i₁ : Quiver.Hom X X'
      i₂ : Quiver.Hom Y Y'
      i₃ : Quiver.Hom S S'
      e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₃) (CategoryTheory.CategoryStru …
      e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
      inst✝ : CategoryTheory.Mono i₃
      z : ↑↑(CategoryTheory.Limits.pullback f' g').toPresheafedSpace
      x : ↑↑X.toPresheafedSpace
      hx : Eq (i₁.base x) ((CategoryTheory.Limits.pullback.fst f' g').base z)
      y : ↑↑Y.toPresheafedSpace
      hy : Eq (i₂.base y) ((CategoryTheory.Limits.pullback.snd f' g').base z)
      ⊢ Membership.mem (Set.range ⇑(CategoryTheory.Limits.pullback.map f g f' g' i₁  …
    -/
    let T₁ : Triplet (pullback.fst f' g') i₁ := Triplet.mk' z x hx.symm
    /-
      case h.mpr
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      X' Y' S' : AlgebraicGeometry.Scheme
      f' : Quiver.Hom X' S'
      g' : Quiver.Hom Y' S'
      i₁ : Quiver.Hom X X'
      i₂ : Quiver.Hom Y Y'
      i₃ : Quiver.Hom S S'
      e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₃) (CategoryTheory.CategoryStru …
      e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
      inst✝ : CategoryTheory.Mono i₃
      z : ↑↑(CategoryTheory.Limits.pullback f' g').toPresheafedSpace
      x : ↑↑X.toPresheafedSpace
      hx : Eq (i₁.base x) ((CategoryTheory.Limits.pullback.fst f' g').base z)
      y : ↑↑Y.toPresheafedSpace
      hy : Eq (i₂.base y) ((CategoryTheory.Limits.pullback.snd f' g').base z)
      T₁ : AlgebraicGeometry.Scheme.Pullback.Triplet (CategoryTheory.Limits.pullback …
      ⊢ Membership.mem (Set.range ⇑(CategoryTheory.Limits.pullback.map f g f' g' i₁  …
    -/
    obtain ⟨w₁, hw₁⟩ := T₁.exists_preimage
    /-
      case h.mpr.intro
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      X' Y' S' : AlgebraicGeometry.Scheme
      f' : Quiver.Hom X' S'
      g' : Quiver.Hom Y' S'
      i₁ : Quiver.Hom X X'
      i₂ : Quiver.Hom Y Y'
      i₃ : Quiver.Hom S S'
      e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₃) (CategoryTheory.CategoryStru …
      e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
      inst✝ : CategoryTheory.Mono i₃
      z : ↑↑(CategoryTheory.Limits.pullback f' g').toPresheafedSpace
      x : ↑↑X.toPresheafedSpace
      hx : Eq (i₁.base x) ((CategoryTheory.Limits.pullback.fst f' g').base z)
      y : ↑↑Y.toPresheafedSpace
      hy : Eq (i₂.base y) ((CategoryTheory.Limits.pullback.snd f' g').base z)
      T₁ : AlgebraicGeometry.Scheme.Pullback.Triplet (CategoryTheory.Limits.pullback …
      w₁ : ↑↑(CategoryTheory.Limits.pullback (CategoryTheory.Limits.pullback.fst f'  …
      hw₁ : And (Eq ((CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.pull …
      ⊢ Membership.mem (Set.range ⇑(CategoryTheory.Limits.pullback.map f g f' g' i₁  …
    -/
    let T₂ : Triplet (pullback.snd f' g') i₂ := Triplet.mk' z y hy.symm
    /-
      case h.mpr.intro
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      X' Y' S' : AlgebraicGeometry.Scheme
      f' : Quiver.Hom X' S'
      g' : Quiver.Hom Y' S'
      i₁ : Quiver.Hom X X'
      i₂ : Quiver.Hom Y Y'
      i₃ : Quiver.Hom S S'
      e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₃) (CategoryTheory.CategoryStru …
      e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
      inst✝ : CategoryTheory.Mono i₃
      z : ↑↑(CategoryTheory.Limits.pullback f' g').toPresheafedSpace
      x : ↑↑X.toPresheafedSpace
      hx : Eq (i₁.base x) ((CategoryTheory.Limits.pullback.fst f' g').base z)
      y : ↑↑Y.toPresheafedSpace
      hy : Eq (i₂.base y) ((CategoryTheory.Limits.pullback.snd f' g').base z)
      T₁ : AlgebraicGeometry.Scheme.Pullback.Triplet (CategoryTheory.Limits.pullback …
      w₁ : ↑↑(CategoryTheory.Limits.pullback (CategoryTheory.Limits.pullback.fst f'  …
      hw₁ : And (Eq ((CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.pull …
      T₂ : AlgebraicGeometry.Scheme.Pullback.Triplet (CategoryTheory.Limits.pullback …
      ⊢ Membership.mem (Set.range ⇑(CategoryTheory.Limits.pullback.map f g f' g' i₁  …
    -/
    obtain ⟨w₂, hw₂⟩ := T₂.exists_preimage
    let T : Triplet (pullback.fst (pullback.fst f' g') i₁) (pullback.fst (pullback.snd f' g') i₂) :=
      Triplet.mk' w₁ w₂ <| by simp [hw₁.left, hw₂.left, T₁, T₂]
    /-
      case h.mpr.intro.intro
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      X' Y' S' : AlgebraicGeometry.Scheme
      f' : Quiver.Hom X' S'
      g' : Quiver.Hom Y' S'
      i₁ : Quiver.Hom X X'
      i₂ : Quiver.Hom Y Y'
      i₃ : Quiver.Hom S S'
      e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₃) (CategoryTheory.CategoryStru …
      e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
      inst✝ : CategoryTheory.Mono i₃
      z : ↑↑(CategoryTheory.Limits.pullback f' g').toPresheafedSpace
      x : ↑↑X.toPresheafedSpace
      hx : Eq (i₁.base x) ((CategoryTheory.Limits.pullback.fst f' g').base z)
      y : ↑↑Y.toPresheafedSpace
      hy : Eq (i₂.base y) ((CategoryTheory.Limits.pullback.snd f' g').base z)
      T₁ : AlgebraicGeometry.Scheme.Pullback.Triplet (CategoryTheory.Limits.pullback …
      w₁ : ↑↑(CategoryTheory.Limits.pullback (CategoryTheory.Limits.pullback.fst f'  …
      hw₁ : And (Eq ((CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.pull …
      T₂ : AlgebraicGeometry.Scheme.Pullback.Triplet (CategoryTheory.Limits.pullback …
      w₂ : ↑↑(CategoryTheory.Limits.pullback (CategoryTheory.Limits.pullback.snd f'  …
      hw₂ : And (Eq ((CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.pull …
      T : AlgebraicGeometry.Scheme.Pullback.Triplet (CategoryTheory.Limits.pullback. …
      ⊢ Membership.mem (Set.range ⇑(CategoryTheory.Limits.pullback.map f g f' g' i₁  …
    -/
    obtain ⟨t, _, ht₂⟩ := T.exists_preimage
    /-
      case h.mpr.intro.intro.intro.intro
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      X' Y' S' : AlgebraicGeometry.Scheme
      f' : Quiver.Hom X' S'
      g' : Quiver.Hom Y' S'
      i₁ : Quiver.Hom X X'
      i₂ : Quiver.Hom Y Y'
      i₃ : Quiver.Hom S S'
      e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₃) (CategoryTheory.CategoryStru …
      e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
      inst✝ : CategoryTheory.Mono i₃
      z : ↑↑(CategoryTheory.Limits.pullback f' g').toPresheafedSpace
      x : ↑↑X.toPresheafedSpace
      hx : Eq (i₁.base x) ((CategoryTheory.Limits.pullback.fst f' g').base z)
      y : ↑↑Y.toPresheafedSpace
      hy : Eq (i₂.base y) ((CategoryTheory.Limits.pullback.snd f' g').base z)
      T₁ : AlgebraicGeometry.Scheme.Pullback.Triplet (CategoryTheory.Limits.pullback …
      w₁ : ↑↑(CategoryTheory.Limits.pullback (CategoryTheory.Limits.pullback.fst f'  …
      hw₁ : And (Eq ((CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.pull …
      T₂ : AlgebraicGeometry.Scheme.Pullback.Triplet (CategoryTheory.Limits.pullback …
      w₂ : ↑↑(CategoryTheory.Limits.pullback (CategoryTheory.Limits.pullback.snd f'  …
      hw₂ : And (Eq ((CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.pull …
      T : AlgebraicGeometry.Scheme.Pullback.Triplet (CategoryTheory.Limits.pullback. …
      t : ↑↑(CategoryTheory.Limits.pullback (CategoryTheory.Limits.pullback.fst (Cat …
      left✝ : Eq ((CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.pullbac …
      ht₂ : Eq ((CategoryTheory.Limits.pullback.snd (CategoryTheory.Limits.pullback. …
      ⊢ Membership.mem (Set.range ⇑(CategoryTheory.Limits.pullback.map f g f' g' i₁  …
    -/
    use (pullbackFstFstIso f g f' g' i₁ i₂ i₃ e₁ e₂).hom.base t
    /-
      case h
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      X' Y' S' : AlgebraicGeometry.Scheme
      f' : Quiver.Hom X' S'
      g' : Quiver.Hom Y' S'
      i₁ : Quiver.Hom X X'
      i₂ : Quiver.Hom Y Y'
      i₃ : Quiver.Hom S S'
      e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₃) (CategoryTheory.CategoryStru …
      e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
      inst✝ : CategoryTheory.Mono i₃
      z : ↑↑(CategoryTheory.Limits.pullback f' g').toPresheafedSpace
      x : ↑↑X.toPresheafedSpace
      hx : Eq (i₁.base x) ((CategoryTheory.Limits.pullback.fst f' g').base z)
      y : ↑↑Y.toPresheafedSpace
      hy : Eq (i₂.base y) ((CategoryTheory.Limits.pullback.snd f' g').base z)
      T₁ : AlgebraicGeometry.Scheme.Pullback.Triplet (CategoryTheory.Limits.pullback …
      w₁ : ↑↑(CategoryTheory.Limits.pullback (CategoryTheory.Limits.pullback.fst f'  …
      hw₁ : And (Eq ((CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.pull …
      T₂ : AlgebraicGeometry.Scheme.Pullback.Triplet (CategoryTheory.Limits.pullback …
      w₂ : ↑↑(CategoryTheory.Limits.pullback (CategoryTheory.Limits.pullback.snd f'  …
      hw₂ : And (Eq ((CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.pull …
      T : AlgebraicGeometry.Scheme.Pullback.Triplet (CategoryTheory.Limits.pullback. …
      t : ↑↑(CategoryTheory.Limits.pullback (CategoryTheory.Limits.pullback.fst (Cat …
      left✝ : Eq ((CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.pullbac …
      ht₂ : Eq ((CategoryTheory.Limits.pullback.snd (CategoryTheory.Limits.pullback. …
      ⊢ Eq ((CategoryTheory.Limits.pullback.map f g f' g' i₁ i₂ i₃ e₁ e₂).base ((Cat …
    -/
    rw [pullback_map_eq_pullbackFstFstIso_inv, ← Scheme.comp_base_apply, Iso.hom_inv_id_assoc]
    /-
      case h
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      X' Y' S' : AlgebraicGeometry.Scheme
      f' : Quiver.Hom X' S'
      g' : Quiver.Hom Y' S'
      i₁ : Quiver.Hom X X'
      i₂ : Quiver.Hom Y Y'
      i₃ : Quiver.Hom S S'
      e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₃) (CategoryTheory.CategoryStru …
      e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
      inst✝ : CategoryTheory.Mono i₃
      z : ↑↑(CategoryTheory.Limits.pullback f' g').toPresheafedSpace
      x : ↑↑X.toPresheafedSpace
      hx : Eq (i₁.base x) ((CategoryTheory.Limits.pullback.fst f' g').base z)
      y : ↑↑Y.toPresheafedSpace
      hy : Eq (i₂.base y) ((CategoryTheory.Limits.pullback.snd f' g').base z)
      T₁ : AlgebraicGeometry.Scheme.Pullback.Triplet (CategoryTheory.Limits.pullback …
      w₁ : ↑↑(CategoryTheory.Limits.pullback (CategoryTheory.Limits.pullback.fst f'  …
      hw₁ : And (Eq ((CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.pull …
      T₂ : AlgebraicGeometry.Scheme.Pullback.Triplet (CategoryTheory.Limits.pullback …
      w₂ : ↑↑(CategoryTheory.Limits.pullback (CategoryTheory.Limits.pullback.snd f'  …
      hw₂ : And (Eq ((CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.pull …
      T : AlgebraicGeometry.Scheme.Pullback.Triplet (CategoryTheory.Limits.pullback. …
      t : ↑↑(CategoryTheory.Limits.pullback (CategoryTheory.Limits.pullback.fst (Cat …
      left✝ : Eq ((CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.pullbac …
      ht₂ : Eq ((CategoryTheory.Limits.pullback.snd (CategoryTheory.Limits.pullback. …
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.snd  …
    -/
    simp [ht₂, T, hw₂.left, T₂]
    /-
      🎉 no goals
    -/


instance isJointlySurjectivePreserving (P : MorphismProperty Scheme.{u}) :
    IsJointlySurjectivePreserving P where
  exists_preimage_fst_triplet_of_prop {X Y S} f g _ hg x y hxy := by
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      x✝ : CategoryTheory.Limits.HasPullback f g
      hg : P g
      x : ↑↑X.toPresheafedSpace
      y : ↑↑Y.toPresheafedSpace
      hxy : Eq (f.base x) (g.base y)
      ⊢ Exists fun a => Eq ((CategoryTheory.Limits.pullback.fst f g).base a) x
    -/
    obtain ⟨a, b, h⟩ := Pullback.exists_preimage_pullback x y hxy
    /-
      case intro.intro
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      x✝ : CategoryTheory.Limits.HasPullback f g
      hg : P g
      x : ↑↑X.toPresheafedSpace
      y : ↑↑Y.toPresheafedSpace
      hxy : Eq (f.base x) (g.base y)
      a : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace
      b : Eq ((CategoryTheory.Limits.pullback.fst f g).base a) x
      h : Eq ((CategoryTheory.Limits.pullback.snd f g).base a) y
      ⊢ Exists fun a => Eq ((CategoryTheory.Limits.pullback.fst f g).base a) x
    -/
    use a
    /-
      🎉 no goals
    -/


instance : MorphismProperty.IsStableUnderBaseChange @Surjective := by
  /-
    ⊢ CategoryTheory.MorphismProperty.IsStableUnderBaseChange @AlgebraicGeometry.S …
  -/
  refine .mk' ?_
  /-
    ⊢ ∀ (X Y S : AlgebraicGeometry.Scheme) (f : Quiver.Hom X S) (g : Quiver.Hom Y  …
  -/
  introv hg
  /-
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    inst✝ : CategoryTheory.Limits.HasPullback f g
    hg : AlgebraicGeometry.Surjective g
    ⊢ AlgebraicGeometry.Surjective (CategoryTheory.Limits.pullback.fst f g)
  -/
  simp only [surjective_iff, ← Set.range_eq_univ, Pullback.range_fst] at hg ⊢
  /-
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    inst✝ : CategoryTheory.Limits.HasPullback f g
    hg : Eq (Set.range ⇑g.base) Set.univ
    ⊢ Eq (Set.preimage (⇑f.base) (Set.range ⇑g.base)) Set.univ
  -/
  rw [hg, Set.preimage_univ]
  /-
    🎉 no goals
  -/


