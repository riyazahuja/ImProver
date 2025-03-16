/-- The kernel of a monoid homomorphism as a congruence relation. -/
@[to_additive "The kernel of an `AddMonoid` homomorphism as an additive congruence relation."]
def ker (f : M →* P) : Con M :=
  mulKer f (map_mul f)


/-- The definition of the congruence relation defined by a monoid homomorphism's kernel. -/
@[to_additive (attr := simp) "The definition of the additive congruence relation defined by an
`AddMonoid` homomorphism's kernel."]
theorem ker_rel (f : M →* P) {x y} : ker f x y ↔ f x = f y :=
  Iff.rfl


/-- The natural homomorphism from a monoid to its quotient by a congruence relation. -/
@[to_additive "The natural homomorphism from an `AddMonoid` to its quotient by an additive
congruence relation."]
def mk' : M →* c.Quotient :=
  { toFun := (↑)
    map_one' := rfl
    map_mul' := fun _ _ => rfl }


/-- The kernel of the natural homomorphism from a monoid to its quotient by a congruence
    relation `c` equals `c`. -/
@[to_additive (attr := simp) "The kernel of the natural homomorphism from an `AddMonoid` to its
quotient by an additive congruence relation `c` equals `c`."]
theorem mk'_ker : ker c.mk' = c :=
  ext fun _ _ => c.eq


/-- The natural homomorphism from a monoid to its quotient by a congruence relation is
    surjective. -/
@[to_additive "The natural homomorphism from an `AddMonoid` to its quotient by a congruence
relation is surjective."]
theorem mk'_surjective : Surjective c.mk' :=
  Quotient.mk''_surjective


@[to_additive (attr := simp)]
theorem coe_mk' : (c.mk' : M → c.Quotient) = ((↑) : M → c.Quotient) :=
  rfl

-- Porting note: used to abuse defeq between sets and predicates

@[to_additive]
theorem ker_apply {f : M →* P} {x y} : ker f x y ↔ f x = f y := Iff.rfl


/-- Given a monoid homomorphism `f : N → M` and a congruence relation `c` on `M`, the congruence
    relation induced on `N` by `f` equals the kernel of `c`'s quotient homomorphism composed with
    `f`. -/
@[to_additive "Given an `AddMonoid` homomorphism `f : N → M` and an additive congruence relation
`c` on `M`, the additive congruence relation induced on `N` by `f` equals the kernel of `c`'s
quotient homomorphism composed with `f`."]
theorem comap_eq {f : N →* M} : comap f f.map_mul c = ker (c.mk'.comp f) :=
                                                   /-
                                                     M : Type u_1
                                                     N : Type u_2
                                                     inst✝¹ : MulOneClass M
                                                     inst✝ : MulOneClass N
                                                     c : Con M
                                                     f : MonoidHom N M
                                                     x y : N
                                                     ⊢ Iff (c (f x) (f y)) (Eq (c.mk' (f x)) (c.mk' (f y)))
                                                   -/
  ext fun x y => show c _ _ ↔ c.mk' _ = c.mk' _ by rw [← c.eq]; rfl
                                                                /-
                                                                  🎉 no goals
                                                                -/


/-- The homomorphism on the quotient of a monoid by a congruence relation `c` induced by a
    homomorphism constant on `c`'s equivalence classes. -/
@[to_additive "The homomorphism on the quotient of an `AddMonoid` by an additive congruence
relation `c` induced by a homomorphism constant on `c`'s equivalence classes."]
def lift (H : c ≤ ker f) : c.Quotient →* P where
  toFun x := (Con.liftOn x f) fun _ _ h => H h
                 /-
                   M : Type u_1
                   N : Type u_2
                   P : Type u_3
                   inst✝² : MulOneClass M
                   inst✝¹ : MulOneClass N
                   inst✝ : MulOneClass P
                   c : Con M
                   x y : M
                   f : MonoidHom M P
                   H : LE.le c (Con.ker f)
                   ⊢ Eq ((fun x => Con.liftOn x ⇑f ⋯) 1) 1
                 -/
  map_one' := by rw [← f.map_one]; rfl
                                   /-
                                     🎉 no goals
                                   -/
  map_mul' x y := Con.induction_on₂ x y fun m n => by
    /-
      M : Type u_1
      N : Type u_2
      P : Type u_3
      inst✝² : MulOneClass M
      inst✝¹ : MulOneClass N
      inst✝ : MulOneClass P
      c : Con M
      x✝ y✝ : M
      f : MonoidHom M P
      H : LE.le c (Con.ker f)
      x y : c.Quotient
      m n : M
      ⊢ Eq ({ toFun := fun x => Con.liftOn x ⇑f ⋯, map_one' := ⋯ }.toFun (HMul.hMul  …
    -/
    dsimp only [← coe_mul, Con.liftOn_coe]
    /-
      M : Type u_1
      N : Type u_2
      P : Type u_3
      inst✝² : MulOneClass M
      inst✝¹ : MulOneClass N
      inst✝ : MulOneClass P
      c : Con M
      x✝ y✝ : M
      f : MonoidHom M P
      H : LE.le c (Con.ker f)
      x y : c.Quotient
      m n : M
      ⊢ Eq (f (HMul.hMul m n)) (HMul.hMul (f m) (f n))
    -/
    rw [map_mul]
    /-
      🎉 no goals
    -/


/-- The diagram describing the universal property for quotients of monoids commutes. -/
@[to_additive "The diagram describing the universal property for quotients of `AddMonoid`s
commutes."]
theorem lift_mk' (H : c ≤ ker f) (x) : c.lift f H (c.mk' x) = f x :=
  rfl


/-- The diagram describing the universal property for quotients of monoids commutes. -/
@[to_additive (attr := simp) "The diagram describing the universal property for quotients of
`AddMonoid`s commutes."]
theorem lift_coe (H : c ≤ ker f) (x : M) : c.lift f H x = f x :=
  rfl


/-- The diagram describing the universal property for quotients of monoids commutes. -/
@[to_additive (attr := simp) "The diagram describing the universal property for quotients of
`AddMonoid`s commutes."]
                                                                          /-
                                                                            M : Type u_1
                                                                            P : Type u_3
                                                                            inst✝¹ : MulOneClass M
                                                                            inst✝ : MulOneClass P
                                                                            c : Con M
                                                                            f : MonoidHom M P
                                                                            H : LE.le c (Con.ker f)
                                                                            ⊢ Eq ((c.lift f H).comp c.mk') f
                                                                          -/
theorem lift_comp_mk' (H : c ≤ ker f) : (c.lift f H).comp c.mk' = f := by ext; rfl
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


/-- Given a homomorphism `f` from the quotient of a monoid by a congruence relation, `f` equals the
    homomorphism on the quotient induced by `f` composed with the natural map from the monoid to
    the quotient. -/
@[to_additive (attr := simp) "Given a homomorphism `f` from the quotient of an `AddMonoid` by an
additive congruence relation, `f` equals the homomorphism on the quotient induced by `f` composed
with the natural map from the `AddMonoid` to the quotient."]
theorem lift_apply_mk' (f : c.Quotient →* P) :
                                                            /-
                                                              M : Type u_1
                                                              N : Type u_2
                                                              P : Type u_3
                                                              inst✝² : MulOneClass M
                                                              inst✝¹ : MulOneClass N
                                                              inst✝ : MulOneClass P
                                                              c : Con M
                                                              x✝ y✝ : M
                                                              f✝ : MonoidHom M P
                                                              f : MonoidHom c.Quotient P
                                                              x y : M
                                                              h : c x y
                                                              ⊢ Eq (f ↑x) (f ↑y)
                                                            -/
    (c.lift (f.comp c.mk') fun x y h => show f ↑x = f ↑y by rw [c.eq.2 h]) = f := by
                                                            /-
                                                              🎉 no goals
                                                            -/
  /-
    M : Type u_1
    P : Type u_3
    inst✝¹ : MulOneClass M
    inst✝ : MulOneClass P
    c : Con M
    f : MonoidHom c.Quotient P
    ⊢ Eq (c.lift (f.comp c.mk') ⋯) f
  -/
  ext x; rcases x with ⟨⟩; rfl
                           /-
                             🎉 no goals
                           -/


/-- Homomorphisms on the quotient of a monoid by a congruence relation are equal if they
    are equal on elements that are coercions from the monoid. -/
@[to_additive "Homomorphisms on the quotient of an `AddMonoid` by an additive congruence relation
are equal if they are equal on elements that are coercions from the `AddMonoid`."]
theorem lift_funext (f g : c.Quotient →* P) (h : ∀ a : M, f a = g a) : f = g := by
  /-
    M : Type u_1
    P : Type u_3
    inst✝¹ : MulOneClass M
    inst✝ : MulOneClass P
    c : Con M
    f g : MonoidHom c.Quotient P
    h : ∀ (a : M), Eq (f ↑a) (g ↑a)
    ⊢ Eq f g
  -/
  rw [← lift_apply_mk' f, ← lift_apply_mk' g]
  /-
    M : Type u_1
    P : Type u_3
    inst✝¹ : MulOneClass M
    inst✝ : MulOneClass P
    c : Con M
    f g : MonoidHom c.Quotient P
    h : ∀ (a : M), Eq (f ↑a) (g ↑a)
    ⊢ Eq (c.lift (f.comp c.mk') ⋯) (c.lift (g.comp c.mk') ⋯)
  -/
  congr 1
  /-
    case e_f
    M : Type u_1
    P : Type u_3
    inst✝¹ : MulOneClass M
    inst✝ : MulOneClass P
    c : Con M
    f g : MonoidHom c.Quotient P
    h : ∀ (a : M), Eq (f ↑a) (g ↑a)
    ⊢ Eq (f.comp c.mk') (g.comp c.mk')
  -/
  exact DFunLike.ext_iff.2 h
  /-
    🎉 no goals
  -/


/-- The uniqueness part of the universal property for quotients of monoids. -/
@[to_additive "The uniqueness part of the universal property for quotients of `AddMonoid`s."]
theorem lift_unique (H : c ≤ ker f) (g : c.Quotient →* P) (Hg : g.comp c.mk' = f) :
    g = c.lift f H :=
  (lift_funext g (c.lift f H)) fun x => by
    /-
      M : Type u_1
      P : Type u_3
      inst✝¹ : MulOneClass M
      inst✝ : MulOneClass P
      c : Con M
      f : MonoidHom M P
      H : LE.le c (Con.ker f)
      g : MonoidHom c.Quotient P
      Hg : Eq (g.comp c.mk') f
      x : M
      ⊢ Eq (g ↑x) ((c.lift f H) ↑x)
    -/
    subst f
    /-
      M : Type u_1
      P : Type u_3
      inst✝¹ : MulOneClass M
      inst✝ : MulOneClass P
      c : Con M
      g : MonoidHom c.Quotient P
      x : M
      H : LE.le c (Con.ker (g.comp c.mk'))
      ⊢ Eq (g ↑x) ((c.lift (g.comp c.mk') H) ↑x)
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- Surjective monoid homomorphisms constant on a congruence relation `c`'s equivalence classes
    induce a surjective homomorphism on `c`'s quotient. -/
@[to_additive "Surjective `AddMonoid` homomorphisms constant on an additive congruence
relation `c`'s equivalence classes induce a surjective homomorphism on `c`'s quotient."]
theorem lift_surjective_of_surjective (h : c ≤ ker f) (hf : Surjective f) :
    Surjective (c.lift f h) := fun y =>
  (Exists.elim (hf y)) fun w hw => ⟨w, (lift_mk' h w).symm ▸ hw⟩


/-- Given a monoid homomorphism `f` from `M` to `P`, the kernel of `f` is the unique congruence
    relation on `M` whose induced map from the quotient of `M` to `P` is injective. -/
@[to_additive "Given an `AddMonoid` homomorphism `f` from `M` to `P`, the kernel of `f`
is the unique additive congruence relation on `M` whose induced map from the quotient of `M`
to `P` is injective."]
theorem ker_eq_lift_of_injective (H : c ≤ ker f) (h : Injective (c.lift f H)) : ker f = c :=
  toSetoid_inj <| Setoid.ker_eq_lift_of_injective f H h


/-- The homomorphism induced on the quotient of a monoid by the kernel of a monoid homomorphism. -/
@[to_additive "The homomorphism induced on the quotient of an `AddMonoid` by the kernel
of an `AddMonoid` homomorphism."]
def kerLift : (ker f).Quotient →* P :=
  ((ker f).lift f) fun _ _ => id


/-- The diagram described by the universal property for quotients of monoids, when the congruence
    relation is the kernel of the homomorphism, commutes. -/
@[to_additive (attr := simp) "The diagram described by the universal property for quotients
of `AddMonoid`s, when the additive congruence relation is the kernel of the homomorphism,
commutes."]
theorem kerLift_mk (x : M) : kerLift f x = f x :=
  rfl


/-- A monoid homomorphism `f` induces an injective homomorphism on the quotient by `f`'s kernel. -/
@[to_additive "An `AddMonoid` homomorphism `f` induces an injective homomorphism on the quotient
by `f`'s kernel."]
theorem kerLift_injective (f : M →* P) : Injective (kerLift f) := fun x y =>
  Quotient.inductionOn₂' x y fun _ _ => (ker f).eq.2


/-- Given congruence relations `c, d` on a monoid such that `d` contains `c`, `d`'s quotient
    map induces a homomorphism from the quotient by `c` to the quotient by `d`. -/
@[to_additive "Given additive congruence relations `c, d` on an `AddMonoid` such that `d`
contains `c`, `d`'s quotient map induces a homomorphism from the quotient by `c` to the quotient
by `d`."]
def map (c d : Con M) (h : c ≤ d) : c.Quotient →* d.Quotient :=
  (c.lift d.mk') fun x y hc => show (ker d.mk') x y from (mk'_ker d).symm ▸ h hc


/-- Given congruence relations `c, d` on a monoid such that `d` contains `c`, the definition of
    the homomorphism from the quotient by `c` to the quotient by `d` induced by `d`'s quotient
    map. -/
@[to_additive "Given additive congruence relations `c, d` on an `AddMonoid` such that `d`
contains `c`, the definition of the homomorphism from the quotient by `c` to the quotient by `d`
induced by `d`'s quotient map."]
theorem map_apply {c d : Con M} (h : c ≤ d) (x) :
    c.map d h x = c.lift d.mk' (fun _ _ hc => d.eq.2 <| h hc) x :=
  rfl


