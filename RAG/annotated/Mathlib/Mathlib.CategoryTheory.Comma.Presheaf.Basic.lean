/-- Via the Yoneda lemma, `u : F.obj (op X)` defines a natural transformation `yoneda.obj X ⟶ F`
    and via the element `η.app (op X) u` also a morphism `yoneda.obj X ⟶ A`. This structure
    witnesses the fact that these morphisms from a commutative triangle with `η : F ⟶ A`, i.e.,
    that `yoneda.obj X ⟶ F` lifts to a morphism in `Over A`. -/
structure MakesOverArrow {F : Cᵒᵖ ⥤ Type v} (η : F ⟶ A) {X : C} (s : yoneda.obj X ⟶ A)
    (u : F.obj (op X)) : Prop where
  app : η.app (op X) u = yonedaEquiv s


/-- "Functoriality" of `MakesOverArrow η s` in `η`. -/
lemma map₁ {F G : Cᵒᵖ ⥤ Type v} {η : F ⟶ A} {μ : G ⟶ A} {ε : F ⟶ G}
    (hε : ε ≫ μ = η) {X : C} {s : yoneda.obj X ⟶ A} {u : F.obj (op X)}
    (h : MakesOverArrow η s u) : MakesOverArrow μ s (ε.app _ u) :=
      /-
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        A F G : CategoryTheory.Functor (Opposite C) (Type v)
        η : Quiver.Hom F A
        μ : Quiver.Hom G A
        ε : Quiver.Hom F G
        hε : Eq (CategoryTheory.CategoryStruct.comp ε μ) η
        X : C
        s : Quiver.Hom (CategoryTheory.yoneda.obj X) A
        u : F.obj { unop := X }
        h : CategoryTheory.OverPresheafAux.MakesOverArrow η s u
        ⊢ Eq (μ.app { unop := X } (ε.app { unop := X } u)) (CategoryTheory.yonedaEquiv …
      -/
  ⟨by rw [← elementwise_of% NatTrans.comp_app ε μ, hε, h.app]⟩
      /-
        🎉 no goals
      -/


/-- "Functoriality of `MakesOverArrow η s` in `s`. -/
lemma map₂ {F : Cᵒᵖ ⥤ Type v} {η : F ⟶ A} {X Y : C} (f : X ⟶ Y)
    {s : yoneda.obj X ⟶ A} {t : yoneda.obj Y ⟶ A} (hst : yoneda.map f ≫ t = s)
    {u : F.obj (op Y)} (h : MakesOverArrow η t u) : MakesOverArrow η s (F.map f.op u) :=
      /-
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        A F : CategoryTheory.Functor (Opposite C) (Type v)
        η : Quiver.Hom F A
        X Y : C
        f : Quiver.Hom X Y
        s : Quiver.Hom (CategoryTheory.yoneda.obj X) A
        t : Quiver.Hom (CategoryTheory.yoneda.obj Y) A
        hst : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.yoneda.map f) t) s
        u : F.obj { unop := Y }
        h : CategoryTheory.OverPresheafAux.MakesOverArrow η t u
        ⊢ Eq (η.app { unop := X } (F.map f.op u)) (CategoryTheory.yonedaEquiv s)
      -/
  ⟨by rw [elementwise_of% η.naturality, h.app, yonedaEquiv_naturality, hst]⟩
      /-
        🎉 no goals
      -/


lemma of_arrow {F : Cᵒᵖ ⥤ Type v} {η : F ⟶ A} {X : C} {s : yoneda.obj X ⟶ A}
    {f : yoneda.obj X ⟶ F} (hf : f ≫ η = s) : MakesOverArrow η s (yonedaEquiv f) :=
  ⟨hf ▸ rfl⟩


lemma of_yoneda_arrow {Y : C} {η : yoneda.obj Y ⟶ A} {X : C} {s : yoneda.obj X ⟶ A} {f : X ⟶ Y}
    (hf : yoneda.map f ≫ η = s) : MakesOverArrow η s f := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    A : CategoryTheory.Functor (Opposite C) (Type v)
    Y : C
    η : Quiver.Hom (CategoryTheory.yoneda.obj Y) A
    X : C
    s : Quiver.Hom (CategoryTheory.yoneda.obj X) A
    f : Quiver.Hom X Y
    hf : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.yoneda.map f) η) s
    ⊢ CategoryTheory.OverPresheafAux.MakesOverArrow η s f
  -/
  simpa only [yonedaEquiv_yoneda_map f] using of_arrow hf
  /-
    🎉 no goals
  -/


/-- This is equivalent to the type `Over.mk s ⟶ Over.mk η`, but that lives in the wrong universe.
    However, if `F = yoneda.obj Y` for some `Y`, then (using that the Yoneda embedding is fully
    faithful) we get a good statement, see `OverArrow.costructuredArrowIso`. -/
def OverArrows {F : Cᵒᵖ ⥤ Type v} (η : F ⟶ A) {X : C} (s : yoneda.obj X ⟶ A) : Type v :=
  Subtype (MakesOverArrow η s)


/-- Since `OverArrows η s` can be thought of to contain certain morphisms `yoneda.obj X ⟶ F`, the
    Yoneda lemma yields elements `F.obj (op X)`. -/
def val {F : Cᵒᵖ ⥤ Type v} {η : F ⟶ A} {X : C} {s : yoneda.obj X ⟶ A} :
    OverArrows η s → F.obj (op X) :=
  Subtype.val


@[simp]
lemma val_mk {F : Cᵒᵖ ⥤ Type v} (η : F ⟶ A) {X : C} (s : yoneda.obj X ⟶ A) (u : F.obj (op X))
    (h : MakesOverArrow η s u) : val ⟨u, h⟩ = u :=
  rfl


@[ext]
lemma ext {F : Cᵒᵖ ⥤ Type v} {η : F ⟶ A} {X : C} {s : yoneda.obj X ⟶ A}
    {u v : OverArrows η s} : u.val = v.val → u = v :=
  Subtype.ext


/-- The defining property of `OverArrows.val`. -/
lemma app_val {F : Cᵒᵖ ⥤ Type v} {η : F ⟶ A} {X : C} {s : yoneda.obj X ⟶ A}
    (p : OverArrows η s) : η.app (op X) p.val = yonedaEquiv s :=
  p.prop.app


/-- In the special case `F = yoneda.obj Y`, the element `p.val` for `p : OverArrows η s` is itself
    a morphism `X ⟶ Y`. -/
@[simp]
lemma map_val {Y : C} {η : yoneda.obj Y ⟶ A} {X : C} {s : yoneda.obj X ⟶ A}
    (p : OverArrows η s) : yoneda.map p.val ≫ η = s := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    A : CategoryTheory.Functor (Opposite C) (Type v)
    Y : C
    η : Quiver.Hom (CategoryTheory.yoneda.obj Y) A
    X : C
    s : Quiver.Hom (CategoryTheory.yoneda.obj X) A
    p : CategoryTheory.OverPresheafAux.OverArrows η s
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.yoneda.map p.val) η) s
  -/
  rw [← yonedaEquiv.injective.eq_iff, yonedaEquiv_comp, yonedaEquiv_yoneda_map]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    A : CategoryTheory.Functor (Opposite C) (Type v)
    Y : C
    η : Quiver.Hom (CategoryTheory.yoneda.obj Y) A
    X : C
    s : Quiver.Hom (CategoryTheory.yoneda.obj X) A
    p : CategoryTheory.OverPresheafAux.OverArrows η s
    ⊢ Eq (η.app { unop := Opposite.unop { unop := X } } p.val) (CategoryTheory.yon …
  -/
  simp only [unop_op, p.app_val]
  /-
    🎉 no goals
  -/


/-- Functoriality of `OverArrows η s` in `η`. -/
def map₁ {F G : Cᵒᵖ ⥤ Type v} {η : F ⟶ A} {μ : G ⟶ A} {X : C} {s : yoneda.obj X ⟶ A}
    (u : OverArrows η s) (ε : F ⟶ G) (hε : ε ≫ μ = η) : OverArrows μ s :=
  ⟨ε.app _ u.val, MakesOverArrow.map₁ hε u.2⟩


@[simp]
lemma map₁_val {F G : Cᵒᵖ ⥤ Type v} {η : F ⟶ A} {μ : G ⟶ A} {X : C}
    (s : yoneda.obj X ⟶ A) (u : OverArrows η s) (ε : F ⟶ G) (hε : ε ≫ μ = η) :
    (u.map₁ ε hε).val = ε.app _ u.val :=
  rfl


/-- Functoriality of `OverArrows η s` in `s`. -/
def map₂ {F : Cᵒᵖ ⥤ Type v} {η : F ⟶ A} {X Y : C} {s : yoneda.obj X ⟶ A}
    {t : yoneda.obj Y ⟶ A} (u : OverArrows η t) (f : X ⟶ Y) (hst : yoneda.map f ≫ t = s) :
    OverArrows η s :=
  ⟨F.map f.op u.val, MakesOverArrow.map₂ f hst u.2⟩


@[simp]
lemma map₂_val {F : Cᵒᵖ ⥤ Type v} {η : F ⟶ A} {X Y : C} (f : X ⟶ Y)
    {s : yoneda.obj X ⟶ A} {t : yoneda.obj Y ⟶ A} (hst : yoneda.map f ≫ t = s)
    (u : OverArrows η t) : (u.map₂ f hst).val = F.map f.op u.val :=
  rfl


@[simp]
lemma map₁_map₂ {F G : Cᵒᵖ ⥤ Type v} {η : F ⟶ A} {μ : G ⟶ A} (ε : F ⟶ G)
    (hε : ε ≫ μ = η) {X Y : C} {s : yoneda.obj X ⟶ A} {t : yoneda.obj Y ⟶ A} (f : X ⟶ Y)
    (hf : yoneda.map f ≫ t = s) (u : OverArrows η t) :
    (u.map₁ ε hε).map₂ f hf = (u.map₂ f hf).map₁ ε hε :=
  OverArrows.ext <| (elementwise_of% (ε.naturality f.op).symm) u.val


/-- Construct an element of `OverArrows η s` with `F = yoneda.obj Y` from a suitable morphism
    `f : X ⟶ Y`. -/
def yonedaArrow {Y : C} {η : yoneda.obj Y ⟶ A} {X : C} {s : yoneda.obj X ⟶ A} (f : X ⟶ Y)
    (hf : yoneda.map f ≫ η = s) : OverArrows η s :=
  ⟨f, .of_yoneda_arrow hf⟩


@[simp]
lemma yonedaArrow_val {Y : C} {η : yoneda.obj Y ⟶ A} {X : C} {s : yoneda.obj X ⟶ A} {f : X ⟶ Y}
    (hf : yoneda.map f ≫ η = s) : (yonedaArrow f hf).val = f :=
  rfl


/-- If `η` is also `yoneda`-costructured, then `OverArrows η s` is just morphisms of costructured
    arrows. -/
def costructuredArrowIso (s t : CostructuredArrow yoneda A) : OverArrows s.hom t.hom ≅ t ⟶ s where
                                             /-
                                               C : Type u
                                               inst✝ : CategoryTheory.Category.{v, u} C
                                               A : CategoryTheory.Functor (Opposite C) (Type v)
                                               s t : CategoryTheory.CostructuredArrow CategoryTheory.yoneda A
                                               p : CategoryTheory.OverPresheafAux.OverArrows s.hom t.hom
                                               ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.yoneda.map p.val) s.h …
                                             -/
  hom p := CostructuredArrow.homMk p.val (by aesop_cat)
                                             /-
                                               🎉 no goals
                                             -/
  inv f := yonedaArrow f.left f.w


/-- This is basically just `yoneda.obj η : (Over A)ᵒᵖ ⥤ Type (max u v)` restricted along the
    forgetful functor `CostructuredArrow yoneda A ⥤ Over A`, but done in a way that we land in a
    smaller universe. -/
@[simps]
def restrictedYonedaObj {F : Cᵒᵖ ⥤ Type v} (η : F ⟶ A) :
    (CostructuredArrow yoneda A)ᵒᵖ ⥤ Type v where
  obj s := OverArrows η s.unop.hom
  map f u := u.map₂ f.unop.left f.unop.w


/-- Functoriality of `restrictedYonedaObj η` in `η`. -/
@[simps]
def restrictedYonedaObjMap₁ {F G : Cᵒᵖ ⥤ Type v} {η : F ⟶ A} {μ : G ⟶ A} (ε : F ⟶ G)
    (hε : ε ≫ μ = η) : restrictedYonedaObj η ⟶ restrictedYonedaObj μ where
  app _ u := u.map₁ ε hε


/-- This is basically just `yoneda : Over A ⥤ (Over A)ᵒᵖ ⥤ Type (max u v)` restricted in the second
    argument along the forgetful functor `CostructuredArrow yoneda A ⥤ Over A`, but done in a way
    that we land in a smaller universe.

    This is one direction of the equivalence we're constructing. -/
@[simps]
def restrictedYoneda (A : Cᵒᵖ ⥤ Type v) : Over A ⥤ (CostructuredArrow yoneda A)ᵒᵖ ⥤ Type v where
  obj η := restrictedYonedaObj η.hom
  map ε := restrictedYonedaObjMap₁ ε.left ε.w


/-- Further restricting the functor
    `restrictedYoneda : Over A ⥤ (CostructuredArrow yoneda A)ᵒᵖ ⥤ Type v` along the forgetful
    functor in the first argument recovers the Yoneda embedding
    `CostructuredArrow yoneda A ⥤ (CostructuredArrow yoneda A)ᵒᵖ ⥤ Type v`. This basically follows
    from the fact that the Yoneda embedding on `C` is fully faithful. -/
def toOverYonedaCompRestrictedYoneda (A : Cᵒᵖ ⥤ Type v) :
    CostructuredArrow.toOver yoneda A ⋙ restrictedYoneda A ≅ yoneda :=
  NatIso.ofComponents
                                                                                     /-
                                                                                       C : Type u
                                                                                       inst✝ : CategoryTheory.Category.{v, u} C
                                                                                       A✝ A : CategoryTheory.Functor (Opposite C) (Type v)
                                                                                       s : CategoryTheory.CostructuredArrow CategoryTheory.yoneda A
                                                                                       ⊢ ∀ {X Y : Opposite (CategoryTheory.CostructuredArrow CategoryTheory.yoneda A) …
                                                                                     -/
    (fun s => NatIso.ofComponents (fun _ => OverArrows.costructuredArrowIso _ _) (by aesop_cat))
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          A✝ A : CategoryTheory.Functor (Opposite C) (Type v)
          ⊢ ∀ {X Y : CategoryTheory.CostructuredArrow CategoryTheory.yoneda A} (f : Quiv …
        -/
    (by aesop_cat)
        /-
          🎉 no goals
        -/


/-- This lemma will be key to establishing good simp normal forms. -/
lemma map_mkPrecomp_eqToHom {F : (CostructuredArrow yoneda A)ᵒᵖ ⥤ Type v} {X Y : C} {f : X ⟶ Y}
    {g g' : yoneda.obj Y ⟶ A} (h : g = g') {x : F.obj (op (CostructuredArrow.mk g'))} :
                                                                   /-
                                                                     C : Type u
                                                                     inst✝ : CategoryTheory.Category.{v, u} C
                                                                     A : CategoryTheory.Functor (Opposite C) (Type v)
                                                                     F : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categor …
                                                                     X Y : C
                                                                     f : Quiver.Hom X Y
                                                                     g g' : Quiver.Hom (CategoryTheory.yoneda.obj Y) A
                                                                     h : Eq g g'
                                                                     x : F.obj { unop := CategoryTheory.CostructuredArrow.mk g' }
                                                                     ⊢ Eq { unop := CategoryTheory.CostructuredArrow.mk g' } { unop := CategoryTheo …
                                                                   -/
    F.map (CostructuredArrow.mkPrecomp g f).op (F.map (eqToHom (by rw [h])) x) =
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
                         /-
                           C : Type u
                           inst✝ : CategoryTheory.Category.{v, u} C
                           A : CategoryTheory.Functor (Opposite C) (Type v)
                           F : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categor …
                           X Y : C
                           f : Quiver.Hom X Y
                           g g' : Quiver.Hom (CategoryTheory.yoneda.obj Y) A
                           h : Eq g g'
                           x : F.obj { unop := CategoryTheory.CostructuredArrow.mk g' }
                           ⊢ Eq { unop := CategoryTheory.CostructuredArrow.mk (CategoryTheory.CategoryStr …
                         -/
      F.map (eqToHom (by rw [h])) (F.map (CostructuredArrow.mkPrecomp g' f).op x) := by
                         /-
                           🎉 no goals
                         -/
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    A : CategoryTheory.Functor (Opposite C) (Type v)
    F : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categor …
    X Y : C
    f : Quiver.Hom X Y
    g g' : Quiver.Hom (CategoryTheory.yoneda.obj Y) A
    h : Eq g g'
    x : F.obj { unop := CategoryTheory.CostructuredArrow.mk g' }
    ⊢ Eq (F.map (CategoryTheory.CostructuredArrow.mkPrecomp g f).op (F.map (Catego …
  -/
  aesop_cat
  /-
    🎉 no goals
  -/


/-- To give an object of `Over A`, we will in particular need a presheaf `Cᵒᵖ ⥤ Type v`. This is the
    definition of that presheaf on objects.

    We would prefer to think of this sigma type to be indexed by natural transformations
    `yoneda.obj X ⟶ A` instead of `A.obj (op X)`. These are equivalent by the Yoneda lemma, but
    we cannot use the former because that type lives in the wrong universe. Hence, we will provide
    a lot of API that will enable us to pretend that we are really indexing over
    `yoneda.obj X ⟶ A`. -/
def YonedaCollection (F : (CostructuredArrow yoneda A)ᵒᵖ ⥤ Type v) (X : C) : Type v :=
  Σ s : A.obj (op X), F.obj (op (CostructuredArrow.mk (yonedaEquiv.symm s)))


/-- Given a costructured arrow `s : yoneda.obj X ⟶ A` and an element `x : F.obj s`, construct
    an element of `YonedaCollection F X`. -/
def mk (s : yoneda.obj X ⟶ A) (x : F.obj (op (CostructuredArrow.mk s))) : YonedaCollection F X :=
                                       /-
                                         C : Type u
                                         inst✝ : CategoryTheory.Category.{v, u} C
                                         A : CategoryTheory.Functor (Opposite C) (Type v)
                                         F : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categor …
                                         X : C
                                         s : Quiver.Hom (CategoryTheory.yoneda.obj X) A
                                         x : F.obj { unop := CategoryTheory.CostructuredArrow.mk s }
                                         ⊢ Eq { unop := CategoryTheory.CostructuredArrow.mk s } { unop := CategoryTheor …
                                       -/
  ⟨yonedaEquiv s, F.map (eqToHom <| by rw [Equiv.symm_apply_apply]) x⟩
                                       /-
                                         🎉 no goals
                                       -/


/-- Access the first component of an element of `YonedaCollection F X`. -/
def fst (p : YonedaCollection F X) : yoneda.obj X ⟶ A :=
  yonedaEquiv.symm p.1


/-- Access the second component of an element of `YonedaCollection F X`. -/
def snd (p : YonedaCollection F X) : F.obj (op (CostructuredArrow.mk p.fst)) :=
  p.2


/-- This is a definition because it will be helpful to be able to control precisely when this
    definition is unfolded. -/
def yonedaEquivFst (p : YonedaCollection F X) : A.obj (op X) :=
  yonedaEquiv p.fst


lemma yonedaEquivFst_eq (p : YonedaCollection F X) : p.yonedaEquivFst = yonedaEquiv p.fst :=
  rfl


@[simp]
lemma mk_fst (s : yoneda.obj X ⟶ A) (x : F.obj (op (CostructuredArrow.mk s))) : (mk s x).fst = s :=
  Equiv.apply_symm_apply _ _


@[simp]
lemma mk_snd (s : yoneda.obj X ⟶ A) (x : F.obj (op (CostructuredArrow.mk s))) :
                                        /-
                                          C : Type u
                                          inst✝ : CategoryTheory.Category.{v, u} C
                                          A : CategoryTheory.Functor (Opposite C) (Type v)
                                          F : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categor …
                                          X : C
                                          s : Quiver.Hom (CategoryTheory.yoneda.obj X) A
                                          x : F.obj { unop := CategoryTheory.CostructuredArrow.mk s }
                                          ⊢ Eq { unop := CategoryTheory.CostructuredArrow.mk s } { unop := CategoryTheor …
                                        -/
    (mk s x).snd = F.map (eqToHom <| by rw [YonedaCollection.mk_fst]) x :=
                                        /-
                                          🎉 no goals
                                        -/
  rfl


@[ext (iff := false)]
lemma ext {p q : YonedaCollection F X} (h : p.fst = q.fst)
                               /-
                                 C : Type u
                                 inst✝ : CategoryTheory.Category.{v, u} C
                                 A : CategoryTheory.Functor (Opposite C) (Type v)
                                 F : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categor …
                                 X : C
                                 p q : CategoryTheory.OverPresheafAux.YonedaCollection F X
                                 h : Eq p.fst q.fst
                                 ⊢ Eq { unop := CategoryTheory.CostructuredArrow.mk q.fst } { unop := CategoryT …
                               -/
    (h' : F.map (eqToHom <| by rw [h]) q.snd = p.snd) : p = q := by
                               /-
                                 🎉 no goals
                               -/
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    A : CategoryTheory.Functor (Opposite C) (Type v)
    F : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categor …
    X : C
    p q : CategoryTheory.OverPresheafAux.YonedaCollection F X
    h : Eq p.fst q.fst
    h' : Eq (F.map (CategoryTheory.eqToHom ⋯) q.snd) p.snd
    ⊢ Eq p q
  -/
  rcases p with ⟨p, p'⟩
  /-
    case mk
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    A : CategoryTheory.Functor (Opposite C) (Type v)
    F : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categor …
    X : C
    q : CategoryTheory.OverPresheafAux.YonedaCollection F X
    p : A.obj { unop := X }
    p' : F.obj { unop := CategoryTheory.CostructuredArrow.mk (CategoryTheory.yoned …
    h : Eq (CategoryTheory.OverPresheafAux.YonedaCollection.fst ⟨p, p'⟩) q.fst
    h' : Eq (F.map (CategoryTheory.eqToHom ⋯) q.snd) (CategoryTheory.OverPresheafA …
    ⊢ Eq ⟨p, p'⟩ q
  -/
  rcases q with ⟨q, q'⟩
  /-
    case mk.mk
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    A : CategoryTheory.Functor (Opposite C) (Type v)
    F : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categor …
    X : C
    p : A.obj { unop := X }
    p' : F.obj { unop := CategoryTheory.CostructuredArrow.mk (CategoryTheory.yoned …
    q : A.obj { unop := X }
    q' : F.obj { unop := CategoryTheory.CostructuredArrow.mk (CategoryTheory.yoned …
    h : Eq (CategoryTheory.OverPresheafAux.YonedaCollection.fst ⟨p, p'⟩) (Category …
    h' : Eq (F.map (CategoryTheory.eqToHom ⋯) (CategoryTheory.OverPresheafAux.Yone …
    ⊢ Eq ⟨p, p'⟩ ⟨q, q'⟩
  -/
  obtain rfl : p = q := yonedaEquiv.symm.injective h
  /-
    case mk.mk
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    A : CategoryTheory.Functor (Opposite C) (Type v)
    F : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categor …
    X : C
    p : A.obj { unop := X }
    p' q' : F.obj { unop := CategoryTheory.CostructuredArrow.mk (CategoryTheory.yo …
    h : Eq (CategoryTheory.OverPresheafAux.YonedaCollection.fst ⟨p, p'⟩) (Category …
    h' : Eq (F.map (CategoryTheory.eqToHom ⋯) (CategoryTheory.OverPresheafAux.Yone …
    ⊢ Eq ⟨p, p'⟩ ⟨p, q'⟩
  -/
  exact Sigma.ext rfl (by simpa [snd] using h'.symm)
  /-
    🎉 no goals
  -/


/-- Functoriality of `YonedaCollection F X` in `F`. -/
def map₁ {G : (CostructuredArrow yoneda A)ᵒᵖ ⥤ Type v} (η : F ⟶ G) :
    YonedaCollection F X → YonedaCollection G X :=
  fun p => YonedaCollection.mk p.fst (η.app _ p.snd)


@[simp]
lemma map₁_fst {G : (CostructuredArrow yoneda A)ᵒᵖ ⥤ Type v} (η : F ⟶ G)
    (p : YonedaCollection F X) : (YonedaCollection.map₁ η p).fst = p.fst := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    A : CategoryTheory.Functor (Opposite C) (Type v)
    F : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categor …
    X : C
    G : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categor …
    η : Quiver.Hom F G
    p : CategoryTheory.OverPresheafAux.YonedaCollection F X
    ⊢ Eq (CategoryTheory.OverPresheafAux.YonedaCollection.map₁ η p).fst p.fst
  -/
  simp [map₁]
  /-
    🎉 no goals
  -/


@[simp]
lemma map₁_yonedaEquivFst {G : (CostructuredArrow yoneda A)ᵒᵖ ⥤ Type v} (η : F ⟶ G)
    (p : YonedaCollection F X) :
    (YonedaCollection.map₁ η p).yonedaEquivFst = p.yonedaEquivFst := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    A : CategoryTheory.Functor (Opposite C) (Type v)
    F : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categor …
    X : C
    G : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categor …
    η : Quiver.Hom F G
    p : CategoryTheory.OverPresheafAux.YonedaCollection F X
    ⊢ Eq (CategoryTheory.OverPresheafAux.YonedaCollection.map₁ η p).yonedaEquivFst …
  -/
  simp only [YonedaCollection.yonedaEquivFst_eq, map₁_fst]
  /-
    🎉 no goals
  -/


@[simp]
lemma map₁_snd {G : (CostructuredArrow yoneda A)ᵒᵖ ⥤ Type v} (η : F ⟶ G)
    (p : YonedaCollection F X) : (YonedaCollection.map₁ η p).snd =
                         /-
                           C : Type u
                           inst✝ : CategoryTheory.Category.{v, u} C
                           A : CategoryTheory.Functor (Opposite C) (Type v)
                           F : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categor …
                           X : C
                           G : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categor …
                           η : Quiver.Hom F G
                           p : CategoryTheory.OverPresheafAux.YonedaCollection F X
                           ⊢ Eq { unop := CategoryTheory.CostructuredArrow.mk p.fst } { unop := CategoryT …
                         -/
      G.map (eqToHom (by rw [YonedaCollection.map₁_fst])) (η.app _ p.snd) := by
                         /-
                           🎉 no goals
                         -/
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    A : CategoryTheory.Functor (Opposite C) (Type v)
    F : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categor …
    X : C
    G : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categor …
    η : Quiver.Hom F G
    p : CategoryTheory.OverPresheafAux.YonedaCollection F X
    ⊢ Eq (CategoryTheory.OverPresheafAux.YonedaCollection.map₁ η p).snd (G.map (Ca …
  -/
  simp [map₁]
  /-
    🎉 no goals
  -/


/-- Functoriality of `YonedaCollection F X` in `X`. -/
def map₂ (F : (CostructuredArrow yoneda A)ᵒᵖ ⥤ Type v) {Y : C} (f : X ⟶ Y)
    (p : YonedaCollection F Y) : YonedaCollection F X :=
  YonedaCollection.mk (yoneda.map f ≫ p.fst) <| F.map (CostructuredArrow.mkPrecomp p.fst f).op p.snd


@[simp]
lemma map₂_fst {Y : C} (f : X ⟶ Y) (p : YonedaCollection F Y) :
    (YonedaCollection.map₂ F f p).fst = yoneda.map f ≫ p.fst := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    A : CategoryTheory.Functor (Opposite C) (Type v)
    F : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categor …
    X Y : C
    f : Quiver.Hom X Y
    p : CategoryTheory.OverPresheafAux.YonedaCollection F Y
    ⊢ Eq (CategoryTheory.OverPresheafAux.YonedaCollection.map₂ F f p).fst (Categor …
  -/
  simp [map₂]
  /-
    🎉 no goals
  -/


@[simp]
lemma map₂_yonedaEquivFst {Y : C} (f : X ⟶ Y) (p : YonedaCollection F Y) :
    (YonedaCollection.map₂ F f p).yonedaEquivFst = A.map f.op p.yonedaEquivFst := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    A : CategoryTheory.Functor (Opposite C) (Type v)
    F : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categor …
    X Y : C
    f : Quiver.Hom X Y
    p : CategoryTheory.OverPresheafAux.YonedaCollection F Y
    ⊢ Eq (CategoryTheory.OverPresheafAux.YonedaCollection.map₂ F f p).yonedaEquivF …
  -/
  simp only [YonedaCollection.yonedaEquivFst_eq, map₂_fst, yonedaEquiv_naturality]
  /-
    🎉 no goals
  -/


@[simp]
lemma map₂_snd {Y : C} (f : X ⟶ Y) (p : YonedaCollection F Y) :
    (YonedaCollection.map₂ F f p).snd = F.map ((CostructuredArrow.mkPrecomp p.fst f).op ≫
                  /-
                    C : Type u
                    inst✝ : CategoryTheory.Category.{v, u} C
                    A : CategoryTheory.Functor (Opposite C) (Type v)
                    F : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categor …
                    X Y : C
                    f : Quiver.Hom X Y
                    p : CategoryTheory.OverPresheafAux.YonedaCollection F Y
                    ⊢ Eq { unop := CategoryTheory.CostructuredArrow.mk (CategoryTheory.CategoryStr …
                  -/
      eqToHom (by rw [YonedaCollection.map₂_fst f])) p.snd := by
                  /-
                    🎉 no goals
                  -/
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    A : CategoryTheory.Functor (Opposite C) (Type v)
    F : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categor …
    X Y : C
    f : Quiver.Hom X Y
    p : CategoryTheory.OverPresheafAux.YonedaCollection F Y
    ⊢ Eq (CategoryTheory.OverPresheafAux.YonedaCollection.map₂ F f p).snd (F.map ( …
  -/
  simp [map₂]
  /-
    🎉 no goals
  -/


@[simp]
lemma map₁_id : YonedaCollection.map₁ (𝟙 F) (X := X) = id := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    A : CategoryTheory.Functor (Opposite C) (Type v)
    F : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categor …
    X : C
    ⊢ Eq (CategoryTheory.OverPresheafAux.YonedaCollection.map₁ (CategoryTheory.Cat …
  -/
  aesop_cat
  /-
    🎉 no goals
  -/


@[simp]
lemma map₁_comp {G H : (CostructuredArrow yoneda A)ᵒᵖ ⥤ Type v} (η : F ⟶ G) (μ : G ⟶ H) :
    YonedaCollection.map₁ (η ≫ μ) (X := X) =
      YonedaCollection.map₁ μ (X := X) ∘ YonedaCollection.map₁ η (X := X) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    A : CategoryTheory.Functor (Opposite C) (Type v)
    F : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categor …
    X : C
    G H : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categ …
    η : Quiver.Hom F G
    μ : Quiver.Hom G H
    ⊢ Eq (CategoryTheory.OverPresheafAux.YonedaCollection.map₁ (CategoryTheory.Cat …
  -/
  ext; all_goals simp
       /-
         🎉 no goals
       -/


@[simp]
lemma map₂_id : YonedaCollection.map₂ F (𝟙 X) = id := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    A : CategoryTheory.Functor (Opposite C) (Type v)
    F : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categor …
    X : C
    ⊢ Eq (CategoryTheory.OverPresheafAux.YonedaCollection.map₂ F (CategoryTheory.C …
  -/
  ext; all_goals simp
       /-
         🎉 no goals
       -/


@[simp]
lemma map₂_comp {Y Z : C} (f : X ⟶ Y) (g : Y ⟶ Z) :
    YonedaCollection.map₂ F (f ≫ g) = YonedaCollection.map₂ F f ∘ YonedaCollection.map₂ F g := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    A : CategoryTheory.Functor (Opposite C) (Type v)
    F : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categor …
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.OverPresheafAux.YonedaCollection.map₂ F (CategoryTheory.C …
  -/
  ext; all_goals simp
       /-
         🎉 no goals
       -/


@[simp]
lemma map₁_map₂ {G : (CostructuredArrow yoneda A)ᵒᵖ ⥤ Type v} (η : F ⟶ G) {Y : C} (f : X ⟶ Y)
    (p : YonedaCollection F Y) :
    YonedaCollection.map₂ G f (YonedaCollection.map₁ η p) =
      YonedaCollection.map₁ η (YonedaCollection.map₂ F f p) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    A : CategoryTheory.Functor (Opposite C) (Type v)
    F : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categor …
    X : C
    G : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categor …
    η : Quiver.Hom F G
    Y : C
    f : Quiver.Hom X Y
    p : CategoryTheory.OverPresheafAux.YonedaCollection F Y
    ⊢ Eq (CategoryTheory.OverPresheafAux.YonedaCollection.map₂ G f (CategoryTheory …
  -/
  ext; all_goals simp
       /-
         🎉 no goals
       -/


/-- Given `F : (CostructuredArrow yoneda A)ᵒᵖ ⥤ Type v`, this is the presheaf that is given by
    `YonedaCollection F X` on objects. -/
@[simps]
def yonedaCollectionPresheaf (A : Cᵒᵖ ⥤ Type v) (F : (CostructuredArrow yoneda A)ᵒᵖ ⥤ Type v) :
    Cᵒᵖ ⥤ Type v where
  obj X := YonedaCollection F X.unop
  map f := YonedaCollection.map₂ F f.unop


/-- Functoriality of `yonedaCollectionPresheaf A F` in `F`. -/
@[simps]
def yonedaCollectionPresheafMap₁ {F G : (CostructuredArrow yoneda A)ᵒᵖ ⥤ Type v} (η : F ⟶ G) :
    yonedaCollectionPresheaf A F ⟶ yonedaCollectionPresheaf A G where
  app _ := YonedaCollection.map₁ η
  naturality := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      A : CategoryTheory.Functor (Opposite C) (Type v)
      F G : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categ …
      η : Quiver.Hom F G
      ⊢ ∀ ⦃X Y : Opposite C⦄ (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct …
    -/
    intros
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      A : CategoryTheory.Functor (Opposite C) (Type v)
      F G : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categ …
      η : Quiver.Hom F G
      X✝ Y✝ : Opposite C
      f✝ : Quiver.Hom X✝ Y✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.OverPresheafAux.yone …
    -/
    ext
    /-
      case h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      A : CategoryTheory.Functor (Opposite C) (Type v)
      F G : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categ …
      η : Quiver.Hom F G
      X✝ Y✝ : Opposite C
      f✝ : Quiver.Hom X✝ Y✝
      a✝ : (CategoryTheory.OverPresheafAux.yonedaCollectionPresheaf A F).obj X✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.OverPresheafAux.yone …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- This is the functor `F ↦ X ↦ YonedaCollection F X`. -/
@[simps]
def yonedaCollectionFunctor (A : Cᵒᵖ ⥤ Type v) :
    ((CostructuredArrow yoneda A)ᵒᵖ ⥤ Type v) ⥤ Cᵒᵖ ⥤ Type v where
  obj := yonedaCollectionPresheaf A
  map η := yonedaCollectionPresheafMap₁ η


/-- The Yoneda lemma yields a natural transformation `yonedaCollectionPresheaf A F ⟶ A`. -/
@[simps]
def yonedaCollectionPresheafToA (F : (CostructuredArrow yoneda A)ᵒᵖ ⥤ Type v) :
    yonedaCollectionPresheaf A F ⟶ A where
  app _ := YonedaCollection.yonedaEquivFst


/-- This is the reverse direction of the equivalence we're constructing. -/
@[simps! obj map]
def costructuredArrowPresheafToOver (A : Cᵒᵖ ⥤ Type v) :
    ((CostructuredArrow yoneda A)ᵒᵖ ⥤ Type v) ⥤ Over A :=
                                                                         /-
                                                                           C : Type u
                                                                           inst✝ : CategoryTheory.Category.{v, u} C
                                                                           A✝ A : CategoryTheory.Functor (Opposite C) (Type v)
                                                                           ⊢ ∀ {Y Z : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow  …
                                                                         -/
  (yonedaCollectionFunctor A).toOver _ (yonedaCollectionPresheafToA) (by aesop_cat)
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


/-- Forward direction of the unit. -/
def unitForward {F : Cᵒᵖ ⥤ Type v} (η : F ⟶ A) (X : C) :
    YonedaCollection (restrictedYonedaObj η) X → F.obj (op X) :=
  fun p => p.snd.val


@[simp]
lemma unitForward_naturality₁ {F G : Cᵒᵖ ⥤ Type v} {η : F ⟶ A} {μ : G ⟶ A} (ε : F ⟶ G)
    (hε : ε ≫ μ = η) (X : C) (p : YonedaCollection (restrictedYonedaObj η) X) :
    unitForward μ X (p.map₁ (restrictedYonedaObjMap₁ ε hε)) = ε.app _ (unitForward η X p) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    A F G : CategoryTheory.Functor (Opposite C) (Type v)
    η : Quiver.Hom F A
    μ : Quiver.Hom G A
    ε : Quiver.Hom F G
    hε : Eq (CategoryTheory.CategoryStruct.comp ε μ) η
    X : C
    p : CategoryTheory.OverPresheafAux.YonedaCollection (CategoryTheory.OverPreshe …
    ⊢ Eq (CategoryTheory.OverPresheafAux.unitForward μ X (CategoryTheory.OverPresh …
  -/
  simp [unitForward]
  /-
    🎉 no goals
  -/


@[simp]
lemma unitForward_naturality₂ {F : Cᵒᵖ ⥤ Type v} (η : F ⟶ A) (X Y : C) (f : X ⟶ Y)
    (p : YonedaCollection (restrictedYonedaObj η) Y) :
    unitForward η X (YonedaCollection.map₂ (restrictedYonedaObj η) f p) =
      F.map f.op (unitForward η Y p) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    A F : CategoryTheory.Functor (Opposite C) (Type v)
    η : Quiver.Hom F A
    X Y : C
    f : Quiver.Hom X Y
    p : CategoryTheory.OverPresheafAux.YonedaCollection (CategoryTheory.OverPreshe …
    ⊢ Eq (CategoryTheory.OverPresheafAux.unitForward η X (CategoryTheory.OverPresh …
  -/
  simp [unitForward]
  /-
    🎉 no goals
  -/


@[simp]
lemma app_unitForward {F : Cᵒᵖ ⥤ Type v} (η : F ⟶ A) (X : Cᵒᵖ)
    (p : YonedaCollection (restrictedYonedaObj η) X.unop) :
    η.app X (unitForward η X.unop p) = p.yonedaEquivFst := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    A F : CategoryTheory.Functor (Opposite C) (Type v)
    η : Quiver.Hom F A
    X : Opposite C
    p : CategoryTheory.OverPresheafAux.YonedaCollection (CategoryTheory.OverPreshe …
    ⊢ Eq (η.app X (CategoryTheory.OverPresheafAux.unitForward η (Opposite.unop X)  …
  -/
  simpa [unitForward] using p.snd.app_val
  /-
    🎉 no goals
  -/


/-- Backward direction of the unit. -/
def unitBackward {F : Cᵒᵖ ⥤ Type v} (η : F ⟶ A) (X : C) :
    F.obj (op X) → YonedaCollection (restrictedYonedaObj η) X :=
                                                                      /-
                                                                        C : Type u
                                                                        inst✝ : CategoryTheory.Category.{v, u} C
                                                                        A F : CategoryTheory.Functor (Opposite C) (Type v)
                                                                        η : Quiver.Hom F A
                                                                        X : C
                                                                        x : F.obj { unop := X }
                                                                        ⊢ Eq (η.app { unop := (Opposite.unop { unop := CategoryTheory.CostructuredArro …
                                                                      -/
  fun x => YonedaCollection.mk (yonedaEquiv.symm (η.app _ x)) ⟨x, ⟨by aesop_cat⟩⟩
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


lemma unitForward_unitBackward {F : Cᵒᵖ ⥤ Type v} (η : F ⟶ A) (X : C) :
    unitForward η X ∘ unitBackward η X = id :=
                     /-
                       C : Type u
                       inst✝ : CategoryTheory.Category.{v, u} C
                       A F : CategoryTheory.Functor (Opposite C) (Type v)
                       η : Quiver.Hom F A
                       X : C
                       x : F.obj { unop := X }
                       ⊢ Eq (Function.comp (CategoryTheory.OverPresheafAux.unitForward η X) (Category …
                     -/
  funext fun x => by simp [unitForward, unitBackward]
                     /-
                       🎉 no goals
                     -/


lemma unitBackward_unitForward {F : Cᵒᵖ ⥤ Type v} (η : F ⟶ A) (X : C) :
    unitBackward η X ∘ unitForward η X = id := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    A F : CategoryTheory.Functor (Opposite C) (Type v)
    η : Quiver.Hom F A
    X : C
    ⊢ Eq (Function.comp (CategoryTheory.OverPresheafAux.unitBackward η X) (Categor …
  -/
  refine funext fun p => YonedaCollection.ext ?_ (OverArrows.ext ?_)
    /-
      case refine_1
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      A F : CategoryTheory.Functor (Opposite C) (Type v)
      η : Quiver.Hom F A
      X : C
      p : CategoryTheory.OverPresheafAux.YonedaCollection (CategoryTheory.OverPreshe …
      ⊢ Eq (Function.comp (CategoryTheory.OverPresheafAux.unitBackward η X) (Categor …
    -/
  · simpa [unitForward, unitBackward] using congrArg yonedaEquiv.symm p.snd.app_val
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      A F : CategoryTheory.Functor (Opposite C) (Type v)
      η : Quiver.Hom F A
      X : C
      p : CategoryTheory.OverPresheafAux.YonedaCollection (CategoryTheory.OverPreshe …
      ⊢ Eq (CategoryTheory.OverPresheafAux.OverArrows.val ((CategoryTheory.OverPresh …
    -/
  · simp [unitForward, unitBackward]
    /-
      🎉 no goals
    -/


/-- Intermediate stage of assembling the unit. -/
@[simps]
def unitAuxAuxAux {F : Cᵒᵖ ⥤ Type v} (η : F ⟶ A) (X : C) :
    YonedaCollection (restrictedYonedaObj η) X ≅ F.obj (op X) where
  hom := unitForward η X
  inv := unitBackward η X
  hom_inv_id := unitBackward_unitForward η X
  inv_hom_id := unitForward_unitBackward η X


/-- Intermediate stage of assembling the unit. -/
@[simps!]
def unitAuxAux {F : Cᵒᵖ ⥤ Type v} (η : F ⟶ A) :
    yonedaCollectionPresheaf A (restrictedYonedaObj η) ≅ F :=
                                                            /-
                                                              C : Type u
                                                              inst✝ : CategoryTheory.Category.{v, u} C
                                                              A F : CategoryTheory.Functor (Opposite C) (Type v)
                                                              η : Quiver.Hom F A
                                                              ⊢ ∀ {X Y : Opposite C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct …
                                                            -/
  NatIso.ofComponents (fun X => unitAuxAuxAux η X.unop) (by aesop_cat)
                                                            /-
                                                              🎉 no goals
                                                            -/


/-- Intermediate stage of assembling the unit. -/
@[simps! hom]
def unitAux (η : Over A) : (restrictedYoneda A ⋙ costructuredArrowPresheafToOver A).obj η ≅ η :=
                                    /-
                                      C : Type u
                                      inst✝ : CategoryTheory.Category.{v, u} C
                                      A : CategoryTheory.Functor (Opposite C) (Type v)
                                      η : CategoryTheory.Over A
                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.OverPresheafAux.unitA …
                                    -/
  Over.isoMk (unitAuxAux η.hom) (by aesop_cat)
                                    /-
                                      🎉 no goals
                                    -/


/-- The unit of the equivalence we're constructing. -/
def unit (A : Cᵒᵖ ⥤ Type v) : 𝟭 (Over A) ≅ restrictedYoneda A ⋙ costructuredArrowPresheafToOver A :=
                                              /-
                                                C : Type u
                                                inst✝ : CategoryTheory.Category.{v, u} C
                                                A✝ A : CategoryTheory.Functor (Opposite C) (Type v)
                                                ⊢ ∀ {X Y : CategoryTheory.Over A} (f : Quiver.Hom X Y), Eq (CategoryTheory.Cat …
                                              -/
  Iso.symm <| NatIso.ofComponents unitAux (by aesop_cat)
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
lemma OverArrows.yonedaCollectionPresheafToA_val_fst (s : yoneda.obj X ⟶ A)
    (p : OverArrows (yonedaCollectionPresheafToA F) s) : p.val.fst = s := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    A : CategoryTheory.Functor (Opposite C) (Type v)
    F : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categor …
    X : C
    s : Quiver.Hom (CategoryTheory.yoneda.obj X) A
    p : CategoryTheory.OverPresheafAux.OverArrows (CategoryTheory.OverPresheafAux. …
    ⊢ Eq (CategoryTheory.OverPresheafAux.YonedaCollection.fst p.val) s
  -/
  simpa [YonedaCollection.yonedaEquivFst_eq] using p.app_val
  /-
    🎉 no goals
  -/


/-- Forward direction of the counit. -/
def counitForward (F : (CostructuredArrow yoneda A)ᵒᵖ ⥤ Type v) (s : CostructuredArrow yoneda A) :
    F.obj (op s) → OverArrows (yonedaCollectionPresheafToA F) s.hom :=
                                             /-
                                               C : Type u
                                               inst✝ : CategoryTheory.Category.{v, u} C
                                               A : CategoryTheory.Functor (Opposite C) (Type v)
                                               F✝ : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Catego …
                                               X : C
                                               F : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categor …
                                               s : CategoryTheory.CostructuredArrow CategoryTheory.yoneda A
                                               x : F.obj { unop := s }
                                               ⊢ Eq ((CategoryTheory.OverPresheafAux.yonedaCollectionPresheafToA F).app { uno …
                                             -/
  fun x => ⟨YonedaCollection.mk s.hom x, ⟨by simp [YonedaCollection.yonedaEquivFst_eq]⟩⟩
                                             /-
                                               🎉 no goals
                                             -/


lemma counitForward_val_fst (s : CostructuredArrow yoneda A) (x : F.obj (op s)) :
    (counitForward F s x).val.fst = s.hom := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    A : CategoryTheory.Functor (Opposite C) (Type v)
    F : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categor …
    s : CategoryTheory.CostructuredArrow CategoryTheory.yoneda A
    x : F.obj { unop := s }
    ⊢ Eq (CategoryTheory.OverPresheafAux.YonedaCollection.fst (CategoryTheory.Over …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma counitForward_val_snd (s : CostructuredArrow yoneda A) (x : F.obj (op s)) :
                                                       /-
                                                         C : Type u
                                                         inst✝ : CategoryTheory.Category.{v, u} C
                                                         A : CategoryTheory.Functor (Opposite C) (Type v)
                                                         F : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categor …
                                                         X : C
                                                         s : CategoryTheory.CostructuredArrow CategoryTheory.yoneda A
                                                         x : F.obj { unop := s }
                                                         ⊢ Eq { unop := s } { unop := CategoryTheory.CostructuredArrow.mk (CategoryTheo …
                                                       -/
    (counitForward F s x).val.snd = F.map (eqToHom (by simp [← CostructuredArrow.eq_mk])) x :=
                                                       /-
                                                         🎉 no goals
                                                       -/
  YonedaCollection.mk_snd _ _


@[simp]
lemma counitForward_naturality₁ {G : (CostructuredArrow yoneda A)ᵒᵖ ⥤ Type v} (η : F ⟶ G)
    (s : (CostructuredArrow yoneda A)ᵒᵖ) (x : F.obj s) : counitForward G s.unop (η.app s x) =
                                                                                      /-
                                                                                        C : Type u
                                                                                        inst✝ : CategoryTheory.Category.{v, u} C
                                                                                        A : CategoryTheory.Functor (Opposite C) (Type v)
                                                                                        F : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categor …
                                                                                        X : C
                                                                                        G : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categor …
                                                                                        η : Quiver.Hom F G
                                                                                        s : Opposite (CategoryTheory.CostructuredArrow CategoryTheory.yoneda A)
                                                                                        x : F.obj s
                                                                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.OverPresheafAux.yoned …
                                                                                      -/
      OverArrows.map₁ (counitForward F s.unop x) (yonedaCollectionPresheafMap₁ η) (by aesop_cat) :=
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/
                                             /-
                                               C : Type u
                                               inst✝ : CategoryTheory.Category.{v, u} C
                                               A : CategoryTheory.Functor (Opposite C) (Type v)
                                               F G : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categ …
                                               η : Quiver.Hom F G
                                               s : Opposite (CategoryTheory.CostructuredArrow CategoryTheory.yoneda A)
                                               x : F.obj s
                                               ⊢ Eq (CategoryTheory.OverPresheafAux.YonedaCollection.fst (CategoryTheory.Over …
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
  OverArrows.ext <| YonedaCollection.ext (by simp) (by simp)
                                                       /-
                                                         🎉 no goals
                                                       -/


@[simp]
lemma counitForward_naturality₂ (s t : (CostructuredArrow yoneda A)ᵒᵖ) (f : t ⟶ s) (x : F.obj t) :
    counitForward F s.unop (F.map f x) =
                                                                 /-
                                                                   C : Type u
                                                                   inst✝ : CategoryTheory.Category.{v, u} C
                                                                   A : CategoryTheory.Functor (Opposite C) (Type v)
                                                                   F : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categor …
                                                                   X : C
                                                                   s t : Opposite (CategoryTheory.CostructuredArrow CategoryTheory.yoneda A)
                                                                   f : Quiver.Hom t s
                                                                   x : F.obj t
                                                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.yoneda.map f.unop.lef …
                                                                 -/
      OverArrows.map₂ (counitForward F t.unop x) f.unop.left (by simp) := by
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    A : CategoryTheory.Functor (Opposite C) (Type v)
    F : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categor …
    s t : Opposite (CategoryTheory.CostructuredArrow CategoryTheory.yoneda A)
    f : Quiver.Hom t s
    x : F.obj t
    ⊢ Eq (CategoryTheory.OverPresheafAux.counitForward F (Opposite.unop s) (F.map  …
  -/
  refine OverArrows.ext <| YonedaCollection.ext (by simp) ?_
  have : (CostructuredArrow.mkPrecomp t.unop.hom f.unop.left).op =
      f ≫ eqToHom (by simp [← CostructuredArrow.eq_mk]) := by
    apply Quiver.Hom.unop_inj
    aesop_cat
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    A : CategoryTheory.Functor (Opposite C) (Type v)
    F : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categor …
    s t : Opposite (CategoryTheory.CostructuredArrow CategoryTheory.yoneda A)
    f : Quiver.Hom t s
    x : F.obj t
    this : Eq (CategoryTheory.CostructuredArrow.mkPrecomp (Opposite.unop t).hom f. …
    ⊢ Eq (F.map (CategoryTheory.eqToHom ⋯) (CategoryTheory.OverPresheafAux.YonedaC …
  -/
  aesop_cat
  /-
    🎉 no goals
  -/


/-- Backward direction of the counit. -/
def counitBackward (F : (CostructuredArrow yoneda A)ᵒᵖ ⥤ Type v) (s : CostructuredArrow yoneda A) :
    OverArrows (yonedaCollectionPresheafToA F) s.hom → F.obj (op s) :=
                              /-
                                C : Type u
                                inst✝ : CategoryTheory.Category.{v, u} C
                                A : CategoryTheory.Functor (Opposite C) (Type v)
                                F✝ : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Catego …
                                X : C
                                F : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categor …
                                s : CategoryTheory.CostructuredArrow CategoryTheory.yoneda A
                                p : CategoryTheory.OverPresheafAux.OverArrows (CategoryTheory.OverPresheafAux. …
                                ⊢ Eq { unop := CategoryTheory.CostructuredArrow.mk (CategoryTheory.OverPreshea …
                              -/
  fun p => F.map (eqToHom (by simp [← CostructuredArrow.eq_mk])) p.val.snd
                              /-
                                🎉 no goals
                              -/


lemma counitForward_counitBackward (F : (CostructuredArrow yoneda A)ᵒᵖ ⥤ Type v)
    (s : CostructuredArrow yoneda A) : counitForward F s ∘ counitBackward F s = id :=
                                                             /-
                                                               C : Type u
                                                               inst✝ : CategoryTheory.Category.{v, u} C
                                                               A : CategoryTheory.Functor (Opposite C) (Type v)
                                                               F : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categor …
                                                               s : CategoryTheory.CostructuredArrow CategoryTheory.yoneda A
                                                               p : CategoryTheory.OverPresheafAux.OverArrows (CategoryTheory.OverPresheafAux. …
                                                               ⊢ Eq (CategoryTheory.OverPresheafAux.YonedaCollection.fst (Function.comp (Cate …
                                                             -/
                                                             /-
                                                               🎉 no goals
                                                             -/
  funext fun p => OverArrows.ext <| YonedaCollection.ext (by simp) (by simp [counitBackward])
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


lemma counitBackward_counitForward (F : (CostructuredArrow yoneda A)ᵒᵖ ⥤ Type v)
    (s : CostructuredArrow yoneda A) : counitBackward F s ∘ counitForward F s = id :=
                     /-
                       C : Type u
                       inst✝ : CategoryTheory.Category.{v, u} C
                       A : CategoryTheory.Functor (Opposite C) (Type v)
                       F : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categor …
                       s : CategoryTheory.CostructuredArrow CategoryTheory.yoneda A
                       x : F.obj { unop := s }
                       ⊢ Eq (Function.comp (CategoryTheory.OverPresheafAux.counitBackward F s) (Categ …
                     -/
  funext fun x => by simp [counitBackward]
                     /-
                       🎉 no goals
                     -/


/-- Intermediate stage of assembling the counit. -/
@[simps]
def counitAuxAux (F : (CostructuredArrow yoneda A)ᵒᵖ ⥤ Type v) (s : CostructuredArrow yoneda A) :
    F.obj (op s) ≅ OverArrows (yonedaCollectionPresheafToA F) s.hom where
  hom := counitForward F s
  inv := counitBackward F s
  hom_inv_id := counitBackward_counitForward F s
  inv_hom_id := counitForward_counitBackward F s


/-- Intermediate stage of assembling the counit. -/
@[simps! hom]
def counitAux (F : (CostructuredArrow yoneda A)ᵒᵖ ⥤ Type v) :
    F ≅ restrictedYonedaObj (yonedaCollectionPresheafToA F) :=
                                                           /-
                                                             C : Type u
                                                             inst✝ : CategoryTheory.Category.{v, u} C
                                                             A : CategoryTheory.Functor (Opposite C) (Type v)
                                                             F✝ : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Catego …
                                                             X : C
                                                             F : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categor …
                                                             ⊢ ∀ {X Y : Opposite (CategoryTheory.CostructuredArrow CategoryTheory.yoneda A) …
                                                           -/
  NatIso.ofComponents (fun s => counitAuxAux F s.unop) (by aesop_cat)
                                                           /-
                                                             🎉 no goals
                                                           -/


/-- The counit of the equivalence we're constructing. -/
def counit (A : Cᵒᵖ ⥤ Type v) : (costructuredArrowPresheafToOver A ⋙ restrictedYoneda A) ≅ 𝟭 _ :=
                                                /-
                                                  C : Type u
                                                  inst✝ : CategoryTheory.Category.{v, u} C
                                                  A✝ : CategoryTheory.Functor (Opposite C) (Type v)
                                                  F : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow Categor …
                                                  X : C
                                                  A : CategoryTheory.Functor (Opposite C) (Type v)
                                                  ⊢ ∀ {X Y : CategoryTheory.Functor (Opposite (CategoryTheory.CostructuredArrow  …
                                                -/
  Iso.symm <| NatIso.ofComponents counitAux (by aesop_cat)
                                                /-
                                                  🎉 no goals
                                                -/


/-- If `A : Cᵒᵖ ⥤ Type v` is a presheaf, then we have an equivalence between presheaves lying over
    `A` and the category of presheaves on `CostructuredArrow yoneda A`. There is a quasicommutative
    triangle involving this equivalence, see
    `CostructuredArrow.toOverCompOverEquivPresheafCostructuredArrow`.

    This is Lemma 1.4.12 in [Kashiwara2006]. -/
def overEquivPresheafCostructuredArrow (A : Cᵒᵖ ⥤ Type v) :
    Over A ≌ ((CostructuredArrow yoneda A)ᵒᵖ ⥤ Type v) :=
  .mk (restrictedYoneda A) (costructuredArrowPresheafToOver A) (unit A) (counit A)


/-- If `A : Cᵒᵖ ⥤ Type v` is a presheaf, then the Yoneda embedding for
    `CostructuredArrow yoneda A` factors through `Over A` via a forgetful functor and an
    equivalence.

    This is Lemma 1.4.12 in [Kashiwara2006]. -/
def CostructuredArrow.toOverCompOverEquivPresheafCostructuredArrow (A : Cᵒᵖ ⥤ Type v) :
    CostructuredArrow.toOver yoneda A ⋙ (overEquivPresheafCostructuredArrow A).functor ≅ yoneda :=
  toOverYonedaCompRestrictedYoneda A


/-- This isomorphism says that hom-sets in the category `Over A` for a presheaf `A` where the domain
    is of the form `(CostructuredArrow.toOver yoneda A).obj X` can instead be interpreted as
    hom-sets in the category `(CostructuredArrow yoneda A)ᵒᵖ ⥤ Type v` where the domain is of the
    form `yoneda.obj X` after adjusting the codomain accordingly. This is desirable because in the
    latter case the Yoneda lemma can be applied. -/
def CostructuredArrow.toOverCompYoneda (A : Cᵒᵖ ⥤ Type v) (T : Over A) :
    (CostructuredArrow.toOver yoneda A).op ⋙ yoneda.obj T ≅
      yoneda.op ⋙ yoneda.obj ((overEquivPresheafCostructuredArrow A).functor.obj T) :=
  NatIso.ofComponents (fun X =>
    (overEquivPresheafCostructuredArrow A).fullyFaithfulFunctor.homEquiv.toIso ≪≫
      (Iso.homCongr
        ((CostructuredArrow.toOverCompOverEquivPresheafCostructuredArrow A).app X.unop)
        (Iso.refl _)).toIso)
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          A✝ A : CategoryTheory.Functor (Opposite C) (Type v)
          T : CategoryTheory.Over A
          ⊢ ∀ {X Y : Opposite (CategoryTheory.CostructuredArrow CategoryTheory.yoneda A) …
        -/
    (by aesop_cat)
        /-
          🎉 no goals
        -/


@[simp]
theorem CostructuredArrow.overEquivPresheafCostructuredArrow_inverse_map_toOverCompYoneda
    {A : Cᵒᵖ ⥤ Type v} {T : Over A} {X : CostructuredArrow yoneda A}
    (f : (CostructuredArrow.toOver yoneda A).obj X ⟶ T) :
    (overEquivPresheafCostructuredArrow A).inverse.map
      (((CostructuredArrow.toOverCompYoneda A T).hom.app (op X) f)) =
      (CostructuredArrow.toOverCompOverEquivPresheafCostructuredArrow A).isoCompInverse.inv.app X ≫
        f ≫ (overEquivPresheafCostructuredArrow A).unit.app T := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    A : CategoryTheory.Functor (Opposite C) (Type v)
    T : CategoryTheory.Over A
    X : CategoryTheory.CostructuredArrow CategoryTheory.yoneda A
    f : Quiver.Hom ((CategoryTheory.CostructuredArrow.toOver CategoryTheory.yoneda …
    ⊢ Eq ((CategoryTheory.overEquivPresheafCostructuredArrow A).inverse.map ((Cate …
  -/
  simp [CostructuredArrow.toOverCompYoneda]
  /-
    🎉 no goals
  -/


@[simp]
theorem CostructuredArrow.overEquivPresheafCostructuredArrow_functor_map_toOverCompYoneda
    {A : Cᵒᵖ ⥤ Type v} {T : Over A} {X : CostructuredArrow yoneda A}
    (f : yoneda.obj X ⟶ (overEquivPresheafCostructuredArrow A).functor.obj T) :
    (overEquivPresheafCostructuredArrow A).functor.map
      (((CostructuredArrow.toOverCompYoneda A T).inv.app (op X) f)) =
      (CostructuredArrow.toOverCompOverEquivPresheafCostructuredArrow A).hom.app X ≫ f := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    A : CategoryTheory.Functor (Opposite C) (Type v)
    T : CategoryTheory.Over A
    X : CategoryTheory.CostructuredArrow CategoryTheory.yoneda A
    f : Quiver.Hom (CategoryTheory.yoneda.obj X) ((CategoryTheory.overEquivPreshea …
    ⊢ Eq ((CategoryTheory.overEquivPresheafCostructuredArrow A).functor.map ((Cate …
  -/
  simp [CostructuredArrow.toOverCompYoneda]
  /-
    🎉 no goals
  -/


/-- This isomorphism says that hom-sets in the category `Over A` for a presheaf `A` where the domain
    is of the form `(CostructuredArrow.toOver yoneda A).obj X` can instead be interpreted as
    hom-sets in the category `(CostructuredArrow yoneda A)ᵒᵖ ⥤ Type v` where the domain is of the
    form `yoneda.obj X` after adjusting the codomain accordingly. This is desirable because in the
    latter case the Yoneda lemma can be applied. -/
def CostructuredArrow.toOverCompCoyoneda (A : Cᵒᵖ ⥤ Type v) :
    (CostructuredArrow.toOver yoneda A).op ⋙ coyoneda ≅
    yoneda.op ⋙ coyoneda ⋙
      (whiskeringLeft _ _ _).obj (overEquivPresheafCostructuredArrow A).functor :=
                                /-
                                  C : Type u
                                  inst✝ : CategoryTheory.Category.{v, u} C
                                  A✝ A : CategoryTheory.Functor (Opposite C) (Type v)
                                  X : Opposite (CategoryTheory.CostructuredArrow CategoryTheory.yoneda A)
                                  ⊢ ∀ {X_1 Y : CategoryTheory.Over A} (f : Quiver.Hom X_1 Y), Eq (CategoryTheory …
                                -/
  NatIso.ofComponents (fun X => NatIso.ofComponents (fun Y =>
                                /-
                                  🎉 no goals
                                -/
    (overEquivPresheafCostructuredArrow A).fullyFaithfulFunctor.homEquiv.toIso ≪≫
      (Iso.homCongr
        ((CostructuredArrow.toOverCompOverEquivPresheafCostructuredArrow A).app X.unop)
                                  /-
                                    C : Type u
                                    inst✝ : CategoryTheory.Category.{v, u} C
                                    A✝ A : CategoryTheory.Functor (Opposite C) (Type v)
                                    ⊢ ∀ {X Y : Opposite (CategoryTheory.CostructuredArrow CategoryTheory.yoneda A) …
                                  -/
        (Iso.refl _)).toIso)) (by aesop_cat)
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
theorem CostructuredArrow.overEquivPresheafCostructuredArrow_inverse_map_toOverCompCoyoneda
    {A : Cᵒᵖ ⥤ Type v} {T : Over A} {X : CostructuredArrow yoneda A}
    (f : (CostructuredArrow.toOver yoneda A).obj X ⟶ T) :
    (overEquivPresheafCostructuredArrow A).inverse.map
      (((CostructuredArrow.toOverCompCoyoneda A).hom.app (op X)).app T f) =
      (CostructuredArrow.toOverCompOverEquivPresheafCostructuredArrow A).isoCompInverse.inv.app X ≫
        f ≫ (overEquivPresheafCostructuredArrow A).unit.app T := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    A : CategoryTheory.Functor (Opposite C) (Type v)
    T : CategoryTheory.Over A
    X : CategoryTheory.CostructuredArrow CategoryTheory.yoneda A
    f : Quiver.Hom ((CategoryTheory.CostructuredArrow.toOver CategoryTheory.yoneda …
    ⊢ Eq ((CategoryTheory.overEquivPresheafCostructuredArrow A).inverse.map (((Cat …
  -/
  simp [CostructuredArrow.toOverCompCoyoneda]
  /-
    🎉 no goals
  -/


@[simp]
theorem CostructuredArrow.overEquivPresheafCostructuredArrow_functor_map_toOverCompCoyoneda
    {A : Cᵒᵖ ⥤ Type v} {T : Over A} {X : CostructuredArrow yoneda A}
    (f : yoneda.obj X ⟶ (overEquivPresheafCostructuredArrow A).functor.obj T) :
    (overEquivPresheafCostructuredArrow A).functor.map
      (((CostructuredArrow.toOverCompCoyoneda A).inv.app (op X)).app T f) =
      (CostructuredArrow.toOverCompOverEquivPresheafCostructuredArrow A).hom.app X ≫ f := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    A : CategoryTheory.Functor (Opposite C) (Type v)
    T : CategoryTheory.Over A
    X : CategoryTheory.CostructuredArrow CategoryTheory.yoneda A
    f : Quiver.Hom (CategoryTheory.yoneda.obj X) ((CategoryTheory.overEquivPreshea …
    ⊢ Eq ((CategoryTheory.overEquivPresheafCostructuredArrow A).functor.map (((Cat …
  -/
  simp [CostructuredArrow.toOverCompCoyoneda]
  /-
    🎉 no goals
  -/


