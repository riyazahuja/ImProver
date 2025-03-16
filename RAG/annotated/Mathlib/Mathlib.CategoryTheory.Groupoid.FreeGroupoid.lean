/-- Shorthand for the "forward" arrow corresponding to `f` in `paths <| symmetrify V` -/
abbrev _root_.Quiver.Hom.toPosPath {X Y : V} (f : X ⟶ Y) :
    (CategoryTheory.Paths.categoryPaths <| Quiver.Symmetrify V).Hom X Y :=
  f.toPos.toPath


/-- Shorthand for the "forward" arrow corresponding to `f` in `paths <| symmetrify V` -/
abbrev _root_.Quiver.Hom.toNegPath {X Y : V} (f : X ⟶ Y) :
    (CategoryTheory.Paths.categoryPaths <| Quiver.Symmetrify V).Hom Y X :=
  f.toNeg.toPath


/-- The "reduction" relation -/
inductive redStep : HomRel (Paths (Quiver.Symmetrify V))
  | step (X Z : Quiver.Symmetrify V) (f : X ⟶ Z) :
    redStep (𝟙 (Paths.of.obj X)) (f.toPath ≫ (Quiver.reverse f).toPath)


/-- The underlying vertices of the free groupoid -/
def _root_.CategoryTheory.FreeGroupoid (V) [Q : Quiver V] :=
  Quotient (@redStep V Q)


instance {V} [Quiver V] [Nonempty V] : Nonempty (FreeGroupoid V) := by
  /-
    V✝ : Type u
    inst✝² : Quiver V✝
    V : Type u_1
    inst✝¹ : Quiver V
    inst✝ : Nonempty V
    ⊢ Nonempty (CategoryTheory.FreeGroupoid V)
  -/
  inhabit V; exact ⟨⟨@default V _⟩⟩
             /-
               🎉 no goals
             -/


theorem congr_reverse {X Y : Paths <| Quiver.Symmetrify V} (p q : X ⟶ Y) :
    Quotient.CompClosure redStep p q → Quotient.CompClosure redStep p.reverse q.reverse := by
  /-
    V : Type u
    inst✝ : Quiver V
    X Y : CategoryTheory.Paths (Quiver.Symmetrify V)
    p q : Quiver.Hom X Y
    ⊢ CategoryTheory.Quotient.CompClosure CategoryTheory.Groupoid.Free.redStep p q …
  -/
  rintro ⟨XW, pp, qq, WY, _, Z, f⟩
  have : Quotient.CompClosure redStep (WY.reverse ≫ 𝟙 _ ≫ XW.reverse)
      (WY.reverse ≫ (f.toPath ≫ (Quiver.reverse f).toPath) ≫ XW.reverse) := by
    constructor
    constructor
  simpa only [CategoryStruct.comp, CategoryStruct.id, Quiver.Path.reverse, Quiver.Path.nil_comp,
    Quiver.Path.reverse_comp, Quiver.reverse_reverse, Quiver.Path.reverse_toPath,
    Quiver.Path.comp_assoc] using this


open Relation in
theorem congr_comp_reverse {X Y : Paths <| Quiver.Symmetrify V} (p : X ⟶ Y) :
    Quot.mk (@Quotient.CompClosure _ _ redStep _ _) (p ≫ p.reverse) =
      Quot.mk (@Quotient.CompClosure _ _ redStep _ _) (𝟙 X) := by
  /-
    V : Type u
    inst✝ : Quiver V
    X Y : CategoryTheory.Paths (Quiver.Symmetrify V)
    p : Quiver.Hom X Y
    ⊢ Eq (Quot.mk (CategoryTheory.Quotient.CompClosure CategoryTheory.Groupoid.Fre …
  -/
  apply Quot.eqvGen_sound
  /-
    case H
    V : Type u
    inst✝ : Quiver V
    X Y : CategoryTheory.Paths (Quiver.Symmetrify V)
    p : Quiver.Hom X Y
    ⊢ Relation.EqvGen (CategoryTheory.Quotient.CompClosure CategoryTheory.Groupoid …
  -/
  induction' p with a b q f ih
    /-
      case H.nil
      V : Type u
      inst✝ : Quiver V
      X Y : CategoryTheory.Paths (Quiver.Symmetrify V)
      ⊢ Relation.EqvGen (CategoryTheory.Quotient.CompClosure CategoryTheory.Groupoid …
    -/
  · apply EqvGen.refl
    /-
      🎉 no goals
    -/
    /-
      case H.cons
      V : Type u
      inst✝ : Quiver V
      X Y a b : CategoryTheory.Paths (Quiver.Symmetrify V)
      q : Quiver.Path X a
      f : Quiver.Hom a b
      ih : Relation.EqvGen (CategoryTheory.Quotient.CompClosure CategoryTheory.Group …
      ⊢ Relation.EqvGen (CategoryTheory.Quotient.CompClosure CategoryTheory.Groupoid …
    -/
  · simp only [Quiver.Path.reverse]
    /-
      case H.cons
      V : Type u
      inst✝ : Quiver V
      X Y a b : CategoryTheory.Paths (Quiver.Symmetrify V)
      q : Quiver.Path X a
      f : Quiver.Hom a b
      ih : Relation.EqvGen (CategoryTheory.Quotient.CompClosure CategoryTheory.Group …
      ⊢ Relation.EqvGen (CategoryTheory.Quotient.CompClosure CategoryTheory.Groupoid …
    -/
    fapply EqvGen.trans
    -- Porting note: `Quiver.Path.*` and `Quiver.Hom.*` notation not working
      /-
        case H.cons.y
        V : Type u
        inst✝ : Quiver V
        X Y a b : CategoryTheory.Paths (Quiver.Symmetrify V)
        q : Quiver.Path X a
        f : Quiver.Hom a b
        ih : Relation.EqvGen (CategoryTheory.Quotient.CompClosure CategoryTheory.Group …
        ⊢ Quiver.Hom X X
      -/
    · exact q ≫ Quiver.Path.reverse q
      /-
        🎉 no goals
      -/
      /-
        case H.cons.a
        V : Type u
        inst✝ : Quiver V
        X Y a b : CategoryTheory.Paths (Quiver.Symmetrify V)
        q : Quiver.Path X a
        f : Quiver.Hom a b
        ih : Relation.EqvGen (CategoryTheory.Quotient.CompClosure CategoryTheory.Group …
        ⊢ Relation.EqvGen (CategoryTheory.Quotient.CompClosure CategoryTheory.Groupoid …
      -/
    · apply EqvGen.symm
      /-
        case H.cons.a.a
        V : Type u
        inst✝ : Quiver V
        X Y a b : CategoryTheory.Paths (Quiver.Symmetrify V)
        q : Quiver.Path X a
        f : Quiver.Hom a b
        ih : Relation.EqvGen (CategoryTheory.Quotient.CompClosure CategoryTheory.Group …
        ⊢ Relation.EqvGen (CategoryTheory.Quotient.CompClosure CategoryTheory.Groupoid …
      -/
      apply EqvGen.rel
      have : Quotient.CompClosure redStep (q ≫ 𝟙 _ ≫ Quiver.Path.reverse q)
          (q ≫ (Quiver.Hom.toPath f ≫ Quiver.Hom.toPath (Quiver.reverse f)) ≫
            Quiver.Path.reverse q) := by
        apply Quotient.CompClosure.intro
        apply redStep.step
      /-
        case H.cons.a.a.a
        V : Type u
        inst✝ : Quiver V
        X Y a b : CategoryTheory.Paths (Quiver.Symmetrify V)
        q : Quiver.Path X a
        f : Quiver.Hom a b
        ih : Relation.EqvGen (CategoryTheory.Quotient.CompClosure CategoryTheory.Group …
        this : CategoryTheory.Quotient.CompClosure CategoryTheory.Groupoid.Free.redSte …
        ⊢ CategoryTheory.Quotient.CompClosure CategoryTheory.Groupoid.Free.redStep (Ca …
      -/
      simp only [Category.assoc, Category.id_comp] at this ⊢
      -- Porting note: `simp` cannot see how `Quiver.Path.comp_assoc` is relevant, so change to
      -- category notation
      change Quotient.CompClosure redStep (q ≫ Quiver.Path.reverse q)
        (Quiver.Path.cons q f ≫ (Quiver.Hom.toPath (Quiver.reverse f)) ≫ (Quiver.Path.reverse q))
      /-
        case H.cons.a.a.a
        V : Type u
        inst✝ : Quiver V
        X Y a b : CategoryTheory.Paths (Quiver.Symmetrify V)
        q : Quiver.Path X a
        f : Quiver.Hom a b
        ih : Relation.EqvGen (CategoryTheory.Quotient.CompClosure CategoryTheory.Group …
        this : CategoryTheory.Quotient.CompClosure CategoryTheory.Groupoid.Free.redSte …
        ⊢ CategoryTheory.Quotient.CompClosure CategoryTheory.Groupoid.Free.redStep (Ca …
      -/
      simp only [← Category.assoc] at this ⊢
      /-
        case H.cons.a.a.a
        V : Type u
        inst✝ : Quiver V
        X Y a b : CategoryTheory.Paths (Quiver.Symmetrify V)
        q : Quiver.Path X a
        f : Quiver.Hom a b
        ih : Relation.EqvGen (CategoryTheory.Quotient.CompClosure CategoryTheory.Group …
        this : CategoryTheory.Quotient.CompClosure CategoryTheory.Groupoid.Free.redSte …
        ⊢ CategoryTheory.Quotient.CompClosure CategoryTheory.Groupoid.Free.redStep (Ca …
      -/
      exact this
      /-
        🎉 no goals
      -/
      /-
        case H.cons.a
        V : Type u
        inst✝ : Quiver V
        X Y a b : CategoryTheory.Paths (Quiver.Symmetrify V)
        q : Quiver.Path X a
        f : Quiver.Hom a b
        ih : Relation.EqvGen (CategoryTheory.Quotient.CompClosure CategoryTheory.Group …
        ⊢ Relation.EqvGen (CategoryTheory.Quotient.CompClosure CategoryTheory.Groupoid …
      -/
    · exact ih
      /-
        🎉 no goals
      -/


theorem congr_reverse_comp {X Y : Paths <| Quiver.Symmetrify V} (p : X ⟶ Y) :
    Quot.mk (@Quotient.CompClosure _ _ redStep _ _) (p.reverse ≫ p) =
      Quot.mk (@Quotient.CompClosure _ _ redStep _ _) (𝟙 Y) := by
  /-
    V : Type u
    inst✝ : Quiver V
    X Y : CategoryTheory.Paths (Quiver.Symmetrify V)
    p : Quiver.Hom X Y
    ⊢ Eq (Quot.mk (CategoryTheory.Quotient.CompClosure CategoryTheory.Groupoid.Fre …
  -/
  nth_rw 2 [← Quiver.Path.reverse_reverse p]
  /-
    V : Type u
    inst✝ : Quiver V
    X Y : CategoryTheory.Paths (Quiver.Symmetrify V)
    p : Quiver.Hom X Y
    ⊢ Eq (Quot.mk (CategoryTheory.Quotient.CompClosure CategoryTheory.Groupoid.Fre …
  -/
  apply congr_comp_reverse
  /-
    🎉 no goals
  -/


instance : Category (FreeGroupoid V) :=
  Quotient.category redStep


/-- The inverse of an arrow in the free groupoid -/
def quotInv {X Y : FreeGroupoid V} (f : X ⟶ Y) : Y ⟶ X :=
  Quot.liftOn f (fun pp => Quot.mk _ <| pp.reverse) fun pp qq con =>
    Quot.sound <| congr_reverse pp qq con


instance _root_.CategoryTheory.FreeGroupoid.instGroupoid : Groupoid (FreeGroupoid V) where
  inv := quotInv
  inv_comp p := Quot.inductionOn p fun pp => congr_reverse_comp pp
  comp_inv p := Quot.inductionOn p fun pp => congr_comp_reverse pp


/-- The inclusion of the quiver on `V` to the underlying quiver on `FreeGroupoid V`-/
def of (V) [Quiver V] : V ⥤q FreeGroupoid V where
  obj X := ⟨X⟩
  map f := Quot.mk _ f.toPosPath


theorem of_eq :
    of V = (Quiver.Symmetrify.of ⋙q Paths.of).comp
      (Quotient.functor <| @redStep V _).toPrefunctor := rfl


/-- The lift of a prefunctor to a groupoid, to a functor from `FreeGroupoid V` -/
def lift (φ : V ⥤q V') : FreeGroupoid V ⥤ V' :=
  Quotient.lift _ (Paths.lift <| Quiver.Symmetrify.lift φ) <| by
    /-
      V : Type u
      inst✝¹ : Quiver V
      V' : Type u'
      inst✝ : CategoryTheory.Groupoid V'
      φ : Prefunctor V V'
      ⊢ ∀ (x y : CategoryTheory.Paths (Quiver.Symmetrify V)) (f₁ f₂ : Quiver.Hom x y …
    -/
    rintro _ _ _ _ ⟨X, Y, f⟩
    -- Porting note: `simp` does not work, so manually `rewrite`
    erw [Paths.lift_nil, Paths.lift_cons, Quiver.Path.comp_nil, Paths.lift_toPath,
      Quiver.Symmetrify.lift_reverse]
    /-
      case step
      V : Type u
      inst✝¹ : Quiver V
      V' : Type u'
      inst✝ : CategoryTheory.Groupoid V'
      φ : Prefunctor V V'
      X✝ Y✝ : CategoryTheory.Paths (Quiver.Symmetrify V)
      X Y : Quiver.Symmetrify V
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.id ((Quiver.Symmetrify.lift φ).obj (Catego …
    -/
    symm
    /-
      case step
      V : Type u
      inst✝¹ : Quiver V
      V' : Type u'
      inst✝ : CategoryTheory.Groupoid V'
      φ : Prefunctor V V'
      X✝ Y✝ : CategoryTheory.Paths (Quiver.Symmetrify V)
      X Y : Quiver.Symmetrify V
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((Quiver.Symmetrify.lift φ).map f) (Q …
    -/
    apply Groupoid.comp_inv
    /-
      🎉 no goals
    -/


theorem lift_spec (φ : V ⥤q V') : of V ⋙q (lift φ).toPrefunctor = φ := by
  /-
    V : Type u
    inst✝¹ : Quiver V
    V' : Type u'
    inst✝ : CategoryTheory.Groupoid V'
    φ : Prefunctor V V'
    ⊢ Eq ((CategoryTheory.Groupoid.Free.of V).comp (CategoryTheory.Groupoid.Free.l …
  -/
  rw [of_eq, Prefunctor.comp_assoc, Prefunctor.comp_assoc, Functor.toPrefunctor_comp]
  /-
    V : Type u
    inst✝¹ : Quiver V
    V' : Type u'
    inst✝ : CategoryTheory.Groupoid V'
    φ : Prefunctor V V'
    ⊢ Eq (Quiver.Symmetrify.of.comp (CategoryTheory.Paths.of.comp ((CategoryTheory …
  -/
  dsimp [lift]
  /-
    V : Type u
    inst✝¹ : Quiver V
    V' : Type u'
    inst✝ : CategoryTheory.Groupoid V'
    φ : Prefunctor V V'
    ⊢ Eq (Quiver.Symmetrify.of.comp (CategoryTheory.Paths.of.comp ((CategoryTheory …
  -/
  rw [Quotient.lift_spec, Paths.lift_spec, Quiver.Symmetrify.lift_spec]
  /-
    🎉 no goals
  -/


theorem lift_unique (φ : V ⥤q V') (Φ : FreeGroupoid V ⥤ V') (hΦ : of V ⋙q Φ.toPrefunctor = φ) :
    Φ = lift φ := by
  /-
    V : Type u
    inst✝¹ : Quiver V
    V' : Type u'
    inst✝ : CategoryTheory.Groupoid V'
    φ : Prefunctor V V'
    Φ : CategoryTheory.Functor (CategoryTheory.FreeGroupoid V) V'
    hΦ : Eq ((CategoryTheory.Groupoid.Free.of V).comp Φ.toPrefunctor) φ
    ⊢ Eq Φ (CategoryTheory.Groupoid.Free.lift φ)
  -/
  apply Quotient.lift_unique
  /-
    case hΦ
    V : Type u
    inst✝¹ : Quiver V
    V' : Type u'
    inst✝ : CategoryTheory.Groupoid V'
    φ : Prefunctor V V'
    Φ : CategoryTheory.Functor (CategoryTheory.FreeGroupoid V) V'
    hΦ : Eq ((CategoryTheory.Groupoid.Free.of V).comp Φ.toPrefunctor) φ
    ⊢ Eq ((CategoryTheory.Quotient.functor CategoryTheory.Groupoid.Free.redStep).c …
  -/
  apply Paths.lift_unique
  /-
    case hΦ.hΦ
    V : Type u
    inst✝¹ : Quiver V
    V' : Type u'
    inst✝ : CategoryTheory.Groupoid V'
    φ : Prefunctor V V'
    Φ : CategoryTheory.Functor (CategoryTheory.FreeGroupoid V) V'
    hΦ : Eq ((CategoryTheory.Groupoid.Free.of V).comp Φ.toPrefunctor) φ
    ⊢ Eq (CategoryTheory.Paths.of.comp ((CategoryTheory.Quotient.functor CategoryT …
  -/
  fapply @Quiver.Symmetrify.lift_unique _ _ _ _ _ _ _ _ _
    /-
      V : Type u
      inst✝¹ : Quiver V
      V' : Type u'
      inst✝ : CategoryTheory.Groupoid V'
      φ : Prefunctor V V'
      Φ : CategoryTheory.Functor (CategoryTheory.FreeGroupoid V) V'
      hΦ : Eq ((CategoryTheory.Groupoid.Free.of V).comp Φ.toPrefunctor) φ
      ⊢ Eq (Quiver.Symmetrify.of.comp (CategoryTheory.Paths.of.comp ((CategoryTheory …
    -/
  · rw [← Functor.toPrefunctor_comp]
    /-
      V : Type u
      inst✝¹ : Quiver V
      V' : Type u'
      inst✝ : CategoryTheory.Groupoid V'
      φ : Prefunctor V V'
      Φ : CategoryTheory.Functor (CategoryTheory.FreeGroupoid V) V'
      hΦ : Eq ((CategoryTheory.Groupoid.Free.of V).comp Φ.toPrefunctor) φ
      ⊢ Eq (Quiver.Symmetrify.of.comp (CategoryTheory.Paths.of.comp ((CategoryTheory …
    -/
    exact hΦ
    /-
      🎉 no goals
    -/
    /-
      V : Type u
      inst✝¹ : Quiver V
      V' : Type u'
      inst✝ : CategoryTheory.Groupoid V'
      φ : Prefunctor V V'
      Φ : CategoryTheory.Functor (CategoryTheory.FreeGroupoid V) V'
      hΦ : Eq ((CategoryTheory.Groupoid.Free.of V).comp Φ.toPrefunctor) φ
      ⊢ ∀ {X Y : Quiver.Symmetrify V} (f : Quiver.Hom X Y), Eq ((CategoryTheory.Path …
    -/
  · rintro X Y f
    /-
      V : Type u
      inst✝¹ : Quiver V
      V' : Type u'
      inst✝ : CategoryTheory.Groupoid V'
      φ : Prefunctor V V'
      Φ : CategoryTheory.Functor (CategoryTheory.FreeGroupoid V) V'
      hΦ : Eq ((CategoryTheory.Groupoid.Free.of V).comp Φ.toPrefunctor) φ
      X Y : Quiver.Symmetrify V
      f : Quiver.Hom X Y
      ⊢ Eq ((CategoryTheory.Paths.of.comp ((CategoryTheory.Quotient.functor Category …
    -/
    simp only [← Functor.toPrefunctor_comp, Prefunctor.comp_map, Paths.of_map, inv_eq_inv]
    change Φ.map (inv ((Quotient.functor redStep).toPrefunctor.map f.toPath)) =
      inv (Φ.map ((Quotient.functor redStep).toPrefunctor.map f.toPath))
    /-
      V : Type u
      inst✝¹ : Quiver V
      V' : Type u'
      inst✝ : CategoryTheory.Groupoid V'
      φ : Prefunctor V V'
      Φ : CategoryTheory.Functor (CategoryTheory.FreeGroupoid V) V'
      hΦ : Eq ((CategoryTheory.Groupoid.Free.of V).comp Φ.toPrefunctor) φ
      X Y : Quiver.Symmetrify V
      f : Quiver.Hom X Y
      ⊢ Eq (Φ.map (CategoryTheory.Groupoid.inv ((CategoryTheory.Quotient.functor Cat …
    -/
    have := Functor.map_inv Φ ((Quotient.functor redStep).toPrefunctor.map f.toPath)
    /-
      V : Type u
      inst✝¹ : Quiver V
      V' : Type u'
      inst✝ : CategoryTheory.Groupoid V'
      φ : Prefunctor V V'
      Φ : CategoryTheory.Functor (CategoryTheory.FreeGroupoid V) V'
      hΦ : Eq ((CategoryTheory.Groupoid.Free.of V).comp Φ.toPrefunctor) φ
      X Y : Quiver.Symmetrify V
      f : Quiver.Hom X Y
      this : Eq (Φ.map (CategoryTheory.inv ((CategoryTheory.Quotient.functor Categor …
      ⊢ Eq (Φ.map (CategoryTheory.Groupoid.inv ((CategoryTheory.Quotient.functor Cat …
    -/
                     /-
                       🎉 no goals
                     -/
    convert this <;> simp only [inv_eq_inv]
                     /-
                       🎉 no goals
                     -/


/-- The functor of free groupoid induced by a prefunctor of quivers -/
def _root_.CategoryTheory.freeGroupoidFunctor (φ : V ⥤q V') : FreeGroupoid V ⥤ FreeGroupoid V' :=
  lift (φ ⋙q of V')


theorem freeGroupoidFunctor_id :
    freeGroupoidFunctor (Prefunctor.id V) = Functor.id (FreeGroupoid V) := by
  /-
    V : Type u
    inst✝ : Quiver V
    ⊢ Eq (CategoryTheory.freeGroupoidFunctor (Prefunctor.id V)) (CategoryTheory.Fu …
  -/
  dsimp only [freeGroupoidFunctor]; symm
  /-
    V : Type u
    inst✝ : Quiver V
    ⊢ Eq (CategoryTheory.Functor.id (CategoryTheory.FreeGroupoid V)) (CategoryTheo …
  -/
  apply lift_unique; rfl
                     /-
                       🎉 no goals
                     -/


theorem freeGroupoidFunctor_comp (φ : V ⥤q V') (φ' : V' ⥤q V'') :
    freeGroupoidFunctor (φ ⋙q φ') = freeGroupoidFunctor φ ⋙ freeGroupoidFunctor φ' := by
  /-
    V : Type u
    inst✝² : Quiver V
    V' : Type u'
    inst✝¹ : Quiver V'
    V'' : Type u''
    inst✝ : Quiver V''
    φ : Prefunctor V V'
    φ' : Prefunctor V' V''
    ⊢ Eq (CategoryTheory.freeGroupoidFunctor (φ.comp φ')) ((CategoryTheory.freeGro …
  -/
  dsimp only [freeGroupoidFunctor]; symm
  /-
    V : Type u
    inst✝² : Quiver V
    V' : Type u'
    inst✝¹ : Quiver V'
    V'' : Type u''
    inst✝ : Quiver V''
    φ : Prefunctor V V'
    φ' : Prefunctor V' V''
    ⊢ Eq ((CategoryTheory.Groupoid.Free.lift (φ.comp (CategoryTheory.Groupoid.Free …
  -/
  apply lift_unique; rfl
                     /-
                       🎉 no goals
                     -/


