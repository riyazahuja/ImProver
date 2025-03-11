/-- A multiplicative action M ↻ X viewed as a functor mapping the single object of M to X
  and an element `m : M` to the map `X → X` given by multiplication by `m`. -/
@[simps]
def actionAsFunctor : SingleObj M ⥤ Type u where
  obj _ := X
  map := (· • ·)
  map_id _ := funext <| MulAction.one_smul
  map_comp f g := funext fun x => (smul_smul g f x).symm


/-- A multiplicative action M ↻ X induces a category structure on X, where a morphism
 from x to y is a scalar taking x to y. Due to implementation details, the object type
 of this category is not equal to X, but is in bijection with X. -/
def ActionCategory :=
  (actionAsFunctor M X).Elements


instance : Category (ActionCategory M X) := by
  /-
    M : Type u_1
    inst✝¹ : Monoid M
    X : Type u
    inst✝ : MulAction M X
    ⊢ CategoryTheory.Category.{?u.1610, u} (CategoryTheory.ActionCategory M X)
  -/
  dsimp only [ActionCategory]
  /-
    M : Type u_1
    inst✝¹ : Monoid M
    X : Type u
    inst✝ : MulAction M X
    ⊢ CategoryTheory.Category.{?u.1610, u} (CategoryTheory.actionAsFunctor M X).El …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The projection from the action category to the monoid, mapping a morphism to its
  label. -/
def π : ActionCategory M X ⥤ SingleObj M :=
  CategoryOfElements.π _


@[simp]
theorem π_map (p q : ActionCategory M X) (f : p ⟶ q) : (π M X).map f = f.val :=
  rfl


@[simp]
theorem π_obj (p : ActionCategory M X) : (π M X).obj p = SingleObj.star M :=
  Unit.ext _ _


/-- The canonical map `ActionCategory M X → X`. It is given by `fun x => x.snd`, but
  has a more explicit type. -/
protected def back : ActionCategory M X → X := fun x => x.snd


instance : CoeTC X (ActionCategory M X) :=
  ⟨fun x => ⟨(), x⟩⟩


@[simp]
theorem coe_back (x : X) : ActionCategory.back (x : ActionCategory M X) = x :=
  rfl


@[simp]
                                                              /-
                                                                M : Type u_1
                                                                inst✝¹ : Monoid M
                                                                X : Type u
                                                                inst✝ : MulAction M X
                                                                x : CategoryTheory.ActionCategory M X
                                                                ⊢ Eq ⟨Unit.unit, x.back⟩ x
                                                              -/
theorem back_coe (x : ActionCategory M X) : ↑x.back = x := by cases x; rfl
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


/-- An object of the action category given by M ↻ X corresponds to an element of X. -/
def objEquiv : X ≃ ActionCategory M X where
  toFun x := x
  invFun x := x.back
  left_inv := coe_back
  right_inv := back_coe


theorem hom_as_subtype (p q : ActionCategory M X) : (p ⟶ q) = { m : M // m • p.back = q.back } :=
  rfl


instance [Inhabited X] : Inhabited (ActionCategory M X) :=
  ⟨show X from default⟩


instance [Nonempty X] : Nonempty (ActionCategory M X) :=
  Nonempty.map (objEquiv M X) inferInstance


/-- The stabilizer of a point is isomorphic to the endomorphism monoid at the
  corresponding point. In fact they are definitionally equivalent. -/
def stabilizerIsoEnd : stabilizerSubmonoid M x ≃* @End (ActionCategory M X) _ x :=
  MulEquiv.refl _


@[simp]
theorem stabilizerIsoEnd_apply (f : stabilizerSubmonoid M x) :
    (stabilizerIsoEnd M x) f = f :=
  rfl


@[simp 1100]
theorem stabilizerIsoEnd_symm_apply (f : End _) : (stabilizerIsoEnd M x).symm f = f :=
  rfl


@[simp]
protected theorem id_val (x : ActionCategory M X) : Subtype.val (𝟙 x) = 1 :=
  rfl


@[simp]
protected theorem comp_val {x y z : ActionCategory M X} (f : x ⟶ y) (g : y ⟶ z) :
    (f ≫ g).val = g.val * f.val :=
  rfl


instance [IsPretransitive M X] [Nonempty X] : IsConnected (ActionCategory M X) :=
  zigzag_isConnected fun x y =>
    Relation.ReflTransGen.single <|
      Or.inl <| nonempty_subtype.mpr (show _ from exists_smul_eq M x.back y.back)


instance : Groupoid (ActionCategory G X) :=
  CategoryTheory.groupoidOfElements _


/-- Any subgroup of `G` is a vertex group in its action groupoid. -/
def endMulEquivSubgroup (H : Subgroup G) : End (objEquiv G (G ⧸ H) ↑(1 : G)) ≃* H :=
  MulEquiv.trans (stabilizerIsoEnd G ((1 : G) : G ⧸ H)).symm
    (MulEquiv.subgroupCongr <| stabilizer_quotient H)


/-- A target vertex `t` and a scalar `g` determine a morphism in the action groupoid. -/
def homOfPair (t : X) (g : G) : @Quiver.Hom (ActionCategory G X) _ (g⁻¹ • t :) t :=
  Subtype.mk g (smul_inv_smul g t)


@[simp]
theorem homOfPair.val (t : X) (g : G) : (homOfPair t g).val = g :=
  rfl


/-- Any morphism in the action groupoid is given by some pair. -/
protected def cases {P : ∀ ⦃a b : ActionCategory G X⦄, (a ⟶ b) → Sort*}
    (hyp : ∀ t g, P (homOfPair t g)) ⦃a b⦄ (f : a ⟶ b) : P f := by
  /-
    M : Type u_1
    inst✝³ : Monoid M
    X : Type u
    inst✝² : MulAction M X
    x : X
    G : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G X
    P : ⦃a b : CategoryTheory.ActionCategory G X⦄ → Quiver.Hom a b → Sort u_3
    hyp : (t : X) → (g : G) → P (CategoryTheory.ActionCategory.homOfPair t g)
    a b : CategoryTheory.ActionCategory G X
    f : Quiver.Hom a b
    ⊢ P f
  -/
  refine cast ?_ (hyp b.back f.val)
  /-
    M : Type u_1
    inst✝³ : Monoid M
    X : Type u
    inst✝² : MulAction M X
    x : X
    G : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G X
    P : ⦃a b : CategoryTheory.ActionCategory G X⦄ → Quiver.Hom a b → Sort u_3
    hyp : (t : X) → (g : G) → P (CategoryTheory.ActionCategory.homOfPair t g)
    a b : CategoryTheory.ActionCategory G X
    f : Quiver.Hom a b
    ⊢ Eq (P (CategoryTheory.ActionCategory.homOfPair b.back ↑f)) (P f)
  -/
  rcases a with ⟨⟨⟩, a : X⟩
  /-
    case mk.unit
    M : Type u_1
    inst✝³ : Monoid M
    X : Type u
    inst✝² : MulAction M X
    x : X
    G : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G X
    P : ⦃a b : CategoryTheory.ActionCategory G X⦄ → Quiver.Hom a b → Sort u_3
    hyp : (t : X) → (g : G) → P (CategoryTheory.ActionCategory.homOfPair t g)
    b : CategoryTheory.ActionCategory G X
    a : X
    f : Quiver.Hom ⟨PUnit.unit, a⟩ b
    ⊢ Eq (P (CategoryTheory.ActionCategory.homOfPair b.back ↑f)) (P f)
  -/
  rcases b with ⟨⟨⟩, b : X⟩
  /-
    case mk.unit.mk.unit
    M : Type u_1
    inst✝³ : Monoid M
    X : Type u
    inst✝² : MulAction M X
    x : X
    G : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G X
    P : ⦃a b : CategoryTheory.ActionCategory G X⦄ → Quiver.Hom a b → Sort u_3
    hyp : (t : X) → (g : G) → P (CategoryTheory.ActionCategory.homOfPair t g)
    a b : X
    f : Quiver.Hom ⟨PUnit.unit, a⟩ ⟨PUnit.unit, b⟩
    ⊢ Eq (P (CategoryTheory.ActionCategory.homOfPair (CategoryTheory.ActionCategor …
  -/
  rcases f with ⟨g : G, h : g • a = b⟩
  /-
    case mk.unit.mk.unit.mk
    M : Type u_1
    inst✝³ : Monoid M
    X : Type u
    inst✝² : MulAction M X
    x : X
    G : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G X
    P : ⦃a b : CategoryTheory.ActionCategory G X⦄ → Quiver.Hom a b → Sort u_3
    hyp : (t : X) → (g : G) → P (CategoryTheory.ActionCategory.homOfPair t g)
    a b : X
    g : G
    h : Eq (HSMul.hSMul g a) b
    ⊢ Eq (P (CategoryTheory.ActionCategory.homOfPair (CategoryTheory.ActionCategor …
  -/
  cases inv_smul_eq_iff.mpr h.symm
  /-
    case mk.unit.mk.unit.mk.refl
    M : Type u_1
    inst✝³ : Monoid M
    X : Type u
    inst✝² : MulAction M X
    x : X
    G : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G X
    P : ⦃a b : CategoryTheory.ActionCategory G X⦄ → Quiver.Hom a b → Sort u_3
    hyp : (t : X) → (g : G) → P (CategoryTheory.ActionCategory.homOfPair t g)
    b : X
    g : G
    h : Eq (HSMul.hSMul g (HSMul.hSMul (Inv.inv g) b)) b
    ⊢ Eq (P (CategoryTheory.ActionCategory.homOfPair (CategoryTheory.ActionCategor …
  -/
  rfl
  /-
    🎉 no goals
  -/

-- Porting note: added to ease the proof of `uncurry`

lemma cases' ⦃a' b' : ActionCategory G X⦄ (f : a' ⟶ b') :
    ∃ (a b : X) (g : G) (ha : a' = a) (hb : b' = b) (hg : a = g⁻¹ • b),
                      /-
                        M : Type u_1
                        inst✝³ : Monoid M
                        X : Type u
                        inst✝² : MulAction M X
                        x : X
                        G : Type u_2
                        inst✝¹ : Group G
                        inst✝ : MulAction G X
                        a' b' : CategoryTheory.ActionCategory G X
                        f : Quiver.Hom a' b'
                        a b : X
                        g : G
                        ha : Eq a' ⟨Unit.unit, a⟩
                        hb : Eq b' ⟨Unit.unit, b⟩
                        hg : Eq a (HSMul.hSMul (Inv.inv g) b)
                        ⊢ Eq a' ⟨Unit.unit, HSMul.hSMul (Inv.inv g) b⟩
                      -/
                      /-
                        🎉 no goals
                      -/
      f = eqToHom (by rw [ha, hg]) ≫ homOfPair b g ≫ eqToHom (by rw [hb]) := by
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
  /-
    X : Type u
    G : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G X
    a' b' : CategoryTheory.ActionCategory G X
    f : Quiver.Hom a' b'
    ⊢ Exists fun a => Exists fun b => Exists fun g => Exists fun ha => Exists fun  …
  -/
  revert a' b' f
  /-
    X : Type u
    G : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G X
    ⊢ ∀ ⦃a' b' : CategoryTheory.ActionCategory G X⦄ (f : Quiver.Hom a' b'), Exists …
  -/
  exact ActionCategory.cases (fun t g => ⟨g⁻¹ • t, t, g, rfl, rfl, rfl, by simp⟩)
  /-
    🎉 no goals
  -/


/-- Given `G` acting on `X`, a functor from the corresponding action groupoid to a group `H`
    can be curried to a group homomorphism `G →* (X → H) ⋊ G`. -/
@[simps]
def curry (F : ActionCategory G X ⥤ SingleObj H) : G →* (X → H) ⋊[mulAutArrow] G :=
  have F_map_eq : ∀ {a b} {f : a ⟶ b}, F.map f = (F.map (homOfPair b.back f.val) : H) := by
    /-
      M : Type u_1
      inst✝⁴ : Monoid M
      X : Type u
      inst✝³ : MulAction M X
      x : X
      G : Type u_2
      inst✝² : Group G
      inst✝¹ : MulAction G X
      H : Type u_3
      inst✝ : Group H
      F : CategoryTheory.Functor (CategoryTheory.ActionCategory G X) (CategoryTheory …
      ⊢ ∀ {a b : CategoryTheory.ActionCategory G X} {f : Quiver.Hom a b}, Eq (F.map  …
    -/
    apply ActionCategory.cases
    /-
      case hyp
      M : Type u_1
      inst✝⁴ : Monoid M
      X : Type u
      inst✝³ : MulAction M X
      x : X
      G : Type u_2
      inst✝² : Group G
      inst✝¹ : MulAction G X
      H : Type u_3
      inst✝ : Group H
      F : CategoryTheory.Functor (CategoryTheory.ActionCategory G X) (CategoryTheory …
      ⊢ ∀ (t : X) (g : G), Eq (F.map (CategoryTheory.ActionCategory.homOfPair t g))  …
    -/
    intros
    /-
      case hyp
      M : Type u_1
      inst✝⁴ : Monoid M
      X : Type u
      inst✝³ : MulAction M X
      x : X
      G : Type u_2
      inst✝² : Group G
      inst✝¹ : MulAction G X
      H : Type u_3
      inst✝ : Group H
      F : CategoryTheory.Functor (CategoryTheory.ActionCategory G X) (CategoryTheory …
      t✝ : X
      g✝ : G
      ⊢ Eq (F.map (CategoryTheory.ActionCategory.homOfPair t✝ g✝)) (F.map (CategoryT …
    -/
    rfl
    /-
      🎉 no goals
    -/
  { toFun := fun g => ⟨fun b => F.map (homOfPair b g), g⟩
    map_one' := by
      /-
        M : Type u_1
        inst✝⁴ : Monoid M
        X : Type u
        inst✝³ : MulAction M X
        x : X
        G : Type u_2
        inst✝² : Group G
        inst✝¹ : MulAction G X
        H : Type u_3
        inst✝ : Group H
        F : CategoryTheory.Functor (CategoryTheory.ActionCategory G X) (CategoryTheory …
        F_map_eq : ∀ {a b : CategoryTheory.ActionCategory G X} {f : Quiver.Hom a b}, E …
        ⊢ Eq ((fun g => { left := fun b => F.map (CategoryTheory.ActionCategory.homOfP …
      -/
      dsimp
      /-
        M : Type u_1
        inst✝⁴ : Monoid M
        X : Type u
        inst✝³ : MulAction M X
        x : X
        G : Type u_2
        inst✝² : Group G
        inst✝¹ : MulAction G X
        H : Type u_3
        inst✝ : Group H
        F : CategoryTheory.Functor (CategoryTheory.ActionCategory G X) (CategoryTheory …
        F_map_eq : ∀ {a b : CategoryTheory.ActionCategory G X} {f : Quiver.Hom a b}, E …
        ⊢ Eq { left := fun b => F.map (CategoryTheory.ActionCategory.homOfPair b 1), r …
      -/
      ext1
        /-
          case left
          M : Type u_1
          inst✝⁴ : Monoid M
          X : Type u
          inst✝³ : MulAction M X
          x : X
          G : Type u_2
          inst✝² : Group G
          inst✝¹ : MulAction G X
          H : Type u_3
          inst✝ : Group H
          F : CategoryTheory.Functor (CategoryTheory.ActionCategory G X) (CategoryTheory …
          F_map_eq : ∀ {a b : CategoryTheory.ActionCategory G X} {f : Quiver.Hom a b}, E …
          ⊢ Eq { left := fun b => F.map (CategoryTheory.ActionCategory.homOfPair b 1), r …
        -/
      · ext b
        /-
          case left.h
          M : Type u_1
          inst✝⁴ : Monoid M
          X : Type u
          inst✝³ : MulAction M X
          x : X
          G : Type u_2
          inst✝² : Group G
          inst✝¹ : MulAction G X
          H : Type u_3
          inst✝ : Group H
          F : CategoryTheory.Functor (CategoryTheory.ActionCategory G X) (CategoryTheory …
          F_map_eq : ∀ {a b : CategoryTheory.ActionCategory G X} {f : Quiver.Hom a b}, E …
          b : X
          ⊢ Eq ({ left := fun b => F.map (CategoryTheory.ActionCategory.homOfPair b 1),  …
        -/
        exact F_map_eq.symm.trans (F.map_id b)
        /-
          🎉 no goals
        -/
      /-
        case right
        M : Type u_1
        inst✝⁴ : Monoid M
        X : Type u
        inst✝³ : MulAction M X
        x : X
        G : Type u_2
        inst✝² : Group G
        inst✝¹ : MulAction G X
        H : Type u_3
        inst✝ : Group H
        F : CategoryTheory.Functor (CategoryTheory.ActionCategory G X) (CategoryTheory …
        F_map_eq : ∀ {a b : CategoryTheory.ActionCategory G X} {f : Quiver.Hom a b}, E …
        ⊢ Eq { left := fun b => F.map (CategoryTheory.ActionCategory.homOfPair b 1), r …
      -/
      rfl
      /-
        🎉 no goals
      -/
    map_mul' := by
      /-
        M : Type u_1
        inst✝⁴ : Monoid M
        X : Type u
        inst✝³ : MulAction M X
        x : X
        G : Type u_2
        inst✝² : Group G
        inst✝¹ : MulAction G X
        H : Type u_3
        inst✝ : Group H
        F : CategoryTheory.Functor (CategoryTheory.ActionCategory G X) (CategoryTheory …
        F_map_eq : ∀ {a b : CategoryTheory.ActionCategory G X} {f : Quiver.Hom a b}, E …
        ⊢ ∀ (x y : G), Eq ({ toFun := fun g => { left := fun b => F.map (CategoryTheor …
      -/
      intro g h
      /-
        M : Type u_1
        inst✝⁴ : Monoid M
        X : Type u
        inst✝³ : MulAction M X
        x : X
        G : Type u_2
        inst✝² : Group G
        inst✝¹ : MulAction G X
        H : Type u_3
        inst✝ : Group H
        F : CategoryTheory.Functor (CategoryTheory.ActionCategory G X) (CategoryTheory …
        F_map_eq : ∀ {a b : CategoryTheory.ActionCategory G X} {f : Quiver.Hom a b}, E …
        g h : G
        ⊢ Eq ({ toFun := fun g => { left := fun b => F.map (CategoryTheory.ActionCateg …
      -/
      ext b
        /-
          case left.h
          M : Type u_1
          inst✝⁴ : Monoid M
          X : Type u
          inst✝³ : MulAction M X
          x : X
          G : Type u_2
          inst✝² : Group G
          inst✝¹ : MulAction G X
          H : Type u_3
          inst✝ : Group H
          F : CategoryTheory.Functor (CategoryTheory.ActionCategory G X) (CategoryTheory …
          F_map_eq : ∀ {a b : CategoryTheory.ActionCategory G X} {f : Quiver.Hom a b}, E …
          g h : G
          b : X
          ⊢ Eq (({ toFun := fun g => { left := fun b => F.map (CategoryTheory.ActionCate …
        -/
      · exact F_map_eq.symm.trans (F.map_comp (homOfPair (g⁻¹ • b) h) (homOfPair b g))
        /-
          🎉 no goals
        -/
      /-
        case right
        M : Type u_1
        inst✝⁴ : Monoid M
        X : Type u
        inst✝³ : MulAction M X
        x : X
        G : Type u_2
        inst✝² : Group G
        inst✝¹ : MulAction G X
        H : Type u_3
        inst✝ : Group H
        F : CategoryTheory.Functor (CategoryTheory.ActionCategory G X) (CategoryTheory …
        F_map_eq : ∀ {a b : CategoryTheory.ActionCategory G X} {f : Quiver.Hom a b}, E …
        g h : G
        ⊢ Eq ({ toFun := fun g => { left := fun b => F.map (CategoryTheory.ActionCateg …
      -/
      rfl }
      /-
        🎉 no goals
      -/


/-- Given `G` acting on `X`, a group homomorphism `φ : G →* (X → H) ⋊ G` can be uncurried to
    a functor from the action groupoid to `H`, provided that `φ g = (_, g)` for all `g`. -/
@[simps]
def uncurry (F : G →* (X → H) ⋊[mulAutArrow] G) (sane : ∀ g, (F g).right = g) :
    ActionCategory G X ⥤ SingleObj H where
  obj _ := ()
  map {_ b} f := (F f.val).left b.back
  map_id x := by
    /-
      M : Type u_1
      inst✝⁴ : Monoid M
      X : Type u
      inst✝³ : MulAction M X
      x✝ : X
      G : Type u_2
      inst✝² : Group G
      inst✝¹ : MulAction G X
      H : Type u_3
      inst✝ : Group H
      F : MonoidHom G (SemidirectProduct (X → H) G mulAutArrow)
      sane : ∀ (g : G), Eq (F g).right g
      x : CategoryTheory.ActionCategory G X
      ⊢ Eq ({ obj := fun x => Unit.unit, map := fun {x b} f => (F ↑f).left b.back }. …
    -/
    dsimp
    /-
      M : Type u_1
      inst✝⁴ : Monoid M
      X : Type u
      inst✝³ : MulAction M X
      x✝ : X
      G : Type u_2
      inst✝² : Group G
      inst✝¹ : MulAction G X
      H : Type u_3
      inst✝ : Group H
      F : MonoidHom G (SemidirectProduct (X → H) G mulAutArrow)
      sane : ∀ (g : G), Eq (F g).right g
      x : CategoryTheory.ActionCategory G X
      ⊢ Eq ((F 1).left x.back) (CategoryTheory.CategoryStruct.id Unit.unit)
    -/
    rw [F.map_one]
    /-
      M : Type u_1
      inst✝⁴ : Monoid M
      X : Type u
      inst✝³ : MulAction M X
      x✝ : X
      G : Type u_2
      inst✝² : Group G
      inst✝¹ : MulAction G X
      H : Type u_3
      inst✝ : Group H
      F : MonoidHom G (SemidirectProduct (X → H) G mulAutArrow)
      sane : ∀ (g : G), Eq (F g).right g
      x : CategoryTheory.ActionCategory G X
      ⊢ Eq (SemidirectProduct.left 1 x.back) (CategoryTheory.CategoryStruct.id Unit. …
    -/
    rfl
    /-
      🎉 no goals
    -/
  map_comp f g := by
    -- Porting note: I was not able to use `ActionCategory.cases` here,
    -- but `ActionCategory.cases'` seems as good; the original proof was:
    -- intro x y z f g; revert y z g
    -- refine' action_category.cases _
    -- simp [single_obj.comp_as_mul, sane]
    /-
      M : Type u_1
      inst✝⁴ : Monoid M
      X : Type u
      inst✝³ : MulAction M X
      x : X
      G : Type u_2
      inst✝² : Group G
      inst✝¹ : MulAction G X
      H : Type u_3
      inst✝ : Group H
      F : MonoidHom G (SemidirectProduct (X → H) G mulAutArrow)
      sane : ∀ (g : G), Eq (F g).right g
      X✝ Y✝ Z✝ : CategoryTheory.ActionCategory G X
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := fun x => Unit.unit, map := fun {x b} f => (F ↑f).left b.back }. …
    -/
    obtain ⟨_, z, γ₁, rfl, rfl, rfl, rfl⟩ := ActionCategory.cases' g
    /-
      case intro.intro.intro.intro.intro.intro
      M : Type u_1
      inst✝⁴ : Monoid M
      X : Type u
      inst✝³ : MulAction M X
      x : X
      G : Type u_2
      inst✝² : Group G
      inst✝¹ : MulAction G X
      H : Type u_3
      inst✝ : Group H
      F : MonoidHom G (SemidirectProduct (X → H) G mulAutArrow)
      sane : ∀ (g : G), Eq (F g).right g
      X✝ : CategoryTheory.ActionCategory G X
      z : X
      γ₁ : G
      f : Quiver.Hom X✝ ⟨Unit.unit, HSMul.hSMul (Inv.inv γ₁) z⟩
      ⊢ Eq ({ obj := fun x => Unit.unit, map := fun {x b} f => (F ↑f).left b.back }. …
    -/
    obtain ⟨_, y, γ₂, rfl, hy, rfl, rfl⟩ := ActionCategory.cases' f
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      M : Type u_1
      inst✝⁴ : Monoid M
      X : Type u
      inst✝³ : MulAction M X
      x : X
      G : Type u_2
      inst✝² : Group G
      inst✝¹ : MulAction G X
      H : Type u_3
      inst✝ : Group H
      F : MonoidHom G (SemidirectProduct (X → H) G mulAutArrow)
      sane : ∀ (g : G), Eq (F g).right g
      z : X
      γ₁ : G
      y : X
      γ₂ : G
      hy : Eq ⟨Unit.unit, HSMul.hSMul (Inv.inv γ₁) z⟩ ⟨Unit.unit, y⟩
      ⊢ Eq ({ obj := fun x => Unit.unit, map := fun {x b} f => (F ↑f).left b.back }. …
    -/
    obtain rfl : y = γ₁⁻¹ • z := congr_arg Sigma.snd hy.symm
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      M : Type u_1
      inst✝⁴ : Monoid M
      X : Type u
      inst✝³ : MulAction M X
      x : X
      G : Type u_2
      inst✝² : Group G
      inst✝¹ : MulAction G X
      H : Type u_3
      inst✝ : Group H
      F : MonoidHom G (SemidirectProduct (X → H) G mulAutArrow)
      sane : ∀ (g : G), Eq (F g).right g
      z : X
      γ₁ γ₂ : G
      hy : Eq ⟨Unit.unit, HSMul.hSMul (Inv.inv γ₁) z⟩ ⟨Unit.unit, HSMul.hSMul (Inv.i …
      ⊢ Eq ({ obj := fun x => Unit.unit, map := fun {x b} f => (F ↑f).left b.back }. …
    -/
    simp [sane]
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      M : Type u_1
      inst✝⁴ : Monoid M
      X : Type u
      inst✝³ : MulAction M X
      x : X
      G : Type u_2
      inst✝² : Group G
      inst✝¹ : MulAction G X
      H : Type u_3
      inst✝ : Group H
      F : MonoidHom G (SemidirectProduct (X → H) G mulAutArrow)
      sane : ∀ (g : G), Eq (F g).right g
      z : X
      γ₁ γ₂ : G
      hy : Eq ⟨Unit.unit, HSMul.hSMul (Inv.inv γ₁) z⟩ ⟨Unit.unit, HSMul.hSMul (Inv.i …
      ⊢ Eq (HMul.hMul ((F γ₁).left z) (HSMul.hSMul γ₁ (F γ₂).left z)) (CategoryTheor …
    -/
    rfl
    /-
      🎉 no goals
    -/


