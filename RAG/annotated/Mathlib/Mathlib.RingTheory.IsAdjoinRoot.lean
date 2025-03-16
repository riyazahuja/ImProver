/-- `IsAdjoinRoot S f` states that the ring `S` can be constructed by adjoining a specified root
of the polynomial `f : R[X]` to `R`.

Compare `PowerBasis R S`, which does not explicitly specify which polynomial we adjoin a root of
(in particular `f` does not need to be the minimal polynomial of the root we adjoin),
and `AdjoinRoot` which constructs a new type.

This is not a typeclass because the choice of root given `S` and `f` is not unique.
-/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): this linter isn't ported yet.
-- @[nolint has_nonempty_instance]
structure IsAdjoinRoot {R : Type u} (S : Type v) [CommSemiring R] [Semiring S] [Algebra R S]
    (f : R[X]) : Type max u v where
  map : R[X] →+* S
  map_surjective : Function.Surjective map
  ker_map : RingHom.ker map = Ideal.span {f}
  algebraMap_eq : algebraMap R S = map.comp Polynomial.C

-- This class doesn't really make sense on a predicate

/-- `IsAdjoinRootMonic S f` states that the ring `S` can be constructed by adjoining a specified
root of the monic polynomial `f : R[X]` to `R`.

As long as `f` is monic, there is a well-defined representation of elements of `S` as polynomials
in `R[X]` of degree lower than `deg f` (see `modByMonicHom` and `coeff`). In particular,
we have `IsAdjoinRootMonic.powerBasis`.

Bundling `Monic` into this structure is very useful when working with explicit `f`s such as
`X^2 - C a * X - C b` since it saves you carrying around the proofs of monicity.
-/
-- @[nolint has_nonempty_instance] -- Porting note: This linter does not exist yet.
structure IsAdjoinRootMonic {R : Type u} (S : Type v) [CommSemiring R] [Semiring S] [Algebra R S]
    (f : R[X]) extends IsAdjoinRoot S f where
  Monic : Monic f


/-- `(h : IsAdjoinRoot S f).root` is the root of `f` that can be adjoined to generate `S`. -/
def root (h : IsAdjoinRoot S f) : S :=
  h.map X


theorem subsingleton (h : IsAdjoinRoot S f) [Subsingleton R] : Subsingleton S :=
  h.map_surjective.subsingleton


theorem algebraMap_apply (h : IsAdjoinRoot S f) (x : R) :
                                                    /-
                                                      R : Type u
                                                      S : Type v
                                                      inst✝² : CommRing R
                                                      inst✝¹ : Ring S
                                                      f : Polynomial R
                                                      inst✝ : Algebra R S
                                                      h : IsAdjoinRoot S f
                                                      x : R
                                                      ⊢ Eq ((algebraMap R S) x) (h.map (Polynomial.C x))
                                                    -/
    algebraMap R S x = h.map (Polynomial.C x) := by rw [h.algebraMap_eq, RingHom.comp_apply]
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem mem_ker_map (h : IsAdjoinRoot S f) {p} : p ∈ RingHom.ker h.map ↔ f ∣ p := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    f : Polynomial R
    inst✝ : Algebra R S
    h : IsAdjoinRoot S f
    p : Polynomial R
    ⊢ Iff (Membership.mem (RingHom.ker h.map) p) (Dvd.dvd f p)
  -/
  rw [h.ker_map, Ideal.mem_span_singleton]
  /-
    🎉 no goals
  -/


@[simp]
theorem map_eq_zero_iff (h : IsAdjoinRoot S f) {p} : h.map p = 0 ↔ f ∣ p := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    f : Polynomial R
    inst✝ : Algebra R S
    h : IsAdjoinRoot S f
    p : Polynomial R
    ⊢ Iff (Eq (h.map p) 0) (Dvd.dvd f p)
  -/
  rw [← h.mem_ker_map, RingHom.mem_ker]
  /-
    🎉 no goals
  -/


@[simp]
theorem map_X (h : IsAdjoinRoot S f) : h.map X = h.root := rfl


@[simp]
theorem map_self (h : IsAdjoinRoot S f) : h.map f = 0 := h.map_eq_zero_iff.mpr dvd_rfl


@[simp]
theorem aeval_eq (h : IsAdjoinRoot S f) (p : R[X]) : aeval h.root p = h.map p :=
                                         /-
                                           R : Type u
                                           S : Type v
                                           inst✝² : CommRing R
                                           inst✝¹ : Ring S
                                           f : Polynomial R
                                           inst✝ : Algebra R S
                                           h : IsAdjoinRoot S f
                                           p : Polynomial R
                                           x : R
                                           ⊢ Eq ((Polynomial.aeval h.root) (Polynomial.C x)) (h.map (Polynomial.C x))
                                         -/
  Polynomial.induction_on p (fun x => by rw [aeval_C, h.algebraMap_apply])
                                         /-
                                           🎉 no goals
                                         -/
                           /-
                             R : Type u
                             S : Type v
                             inst✝² : CommRing R
                             inst✝¹ : Ring S
                             f : Polynomial R
                             inst✝ : Algebra R S
                             h : IsAdjoinRoot S f
                             p✝ p q : Polynomial R
                             ihp : Eq ((Polynomial.aeval h.root) p) (h.map p)
                             ihq : Eq ((Polynomial.aeval h.root) q) (h.map q)
                             ⊢ Eq ((Polynomial.aeval h.root) (HAdd.hAdd p q)) (h.map (HAdd.hAdd p q))
                           -/
    (fun p q ihp ihq => by rw [map_add, RingHom.map_add, ihp, ihq]) fun n x _ => by
                           /-
                             🎉 no goals
                           -/
    rw [map_mul, aeval_C, map_pow, aeval_X, RingHom.map_mul, ← h.algebraMap_apply,
      RingHom.map_pow, map_X]


                                                                     /-
                                                                       R : Type u
                                                                       S : Type v
                                                                       inst✝² : CommRing R
                                                                       inst✝¹ : Ring S
                                                                       f : Polynomial R
                                                                       inst✝ : Algebra R S
                                                                       h : IsAdjoinRoot S f
                                                                       ⊢ Eq ((Polynomial.aeval h.root) f) 0
                                                                     -/
theorem aeval_root (h : IsAdjoinRoot S f) : aeval h.root f = 0 := by rw [aeval_eq, map_self]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


/-- Choose an arbitrary representative so that `h.map (h.repr x) = x`.

If `f` is monic, use `IsAdjoinRootMonic.modByMonicHom` for a unique choice of representative.
-/
def repr (h : IsAdjoinRoot S f) (x : S) : R[X] :=
  (h.map_surjective x).choose


theorem map_repr (h : IsAdjoinRoot S f) (x : S) : h.map (h.repr x) = x :=
  (h.map_surjective x).choose_spec


/-- `repr` preserves zero, up to multiples of `f` -/
theorem repr_zero_mem_span (h : IsAdjoinRoot S f) : h.repr 0 ∈ Ideal.span ({f} : Set R[X]) := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    f : Polynomial R
    inst✝ : Algebra R S
    h : IsAdjoinRoot S f
    ⊢ Membership.mem (Ideal.span (Singleton.singleton f)) (h.repr 0)
  -/
  rw [← h.ker_map, RingHom.mem_ker, h.map_repr]
  /-
    🎉 no goals
  -/


/-- `repr` preserves addition, up to multiples of `f` -/
theorem repr_add_sub_repr_add_repr_mem_span (h : IsAdjoinRoot S f) (x y : S) :
    h.repr (x + y) - (h.repr x + h.repr y) ∈ Ideal.span ({f} : Set R[X]) := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    f : Polynomial R
    inst✝ : Algebra R S
    h : IsAdjoinRoot S f
    x y : S
    ⊢ Membership.mem (Ideal.span (Singleton.singleton f)) (HSub.hSub (h.repr (HAdd …
  -/
  rw [← h.ker_map, RingHom.mem_ker, map_sub, h.map_repr, map_add, h.map_repr, h.map_repr, sub_self]
  /-
    🎉 no goals
  -/


/-- Extensionality of the `IsAdjoinRoot` structure itself. See `IsAdjoinRootMonic.ext_elem`
for extensionality of the ring elements. -/
theorem ext_map (h h' : IsAdjoinRoot S f) (eq : ∀ x, h.map x = h'.map x) : h = h' := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    f : Polynomial R
    inst✝ : Algebra R S
    h h' : IsAdjoinRoot S f
    eq : ∀ (x : Polynomial R), Eq (h.map x) (h'.map x)
    ⊢ Eq h h'
  -/
  cases h; cases h'; congr
  /-
    case mk.mk.e_map
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    f : Polynomial R
    inst✝ : Algebra R S
    map✝¹ : RingHom (Polynomial R) S
    map_surjective✝¹ : Function.Surjective ⇑map✝¹
    ker_map✝¹ : Eq (RingHom.ker map✝¹) (Ideal.span (Singleton.singleton f))
    algebraMap_eq✝¹ : Eq (algebraMap R S) (map✝¹.comp Polynomial.C)
    map✝ : RingHom (Polynomial R) S
    map_surjective✝ : Function.Surjective ⇑map✝
    ker_map✝ : Eq (RingHom.ker map✝) (Ideal.span (Singleton.singleton f))
    algebraMap_eq✝ : Eq (algebraMap R S) (map✝.comp Polynomial.C)
    eq : ∀ (x : Polynomial R), Eq ({ map := map✝¹, map_surjective := map_surjectiv …
    ⊢ Eq map✝¹ map✝
  -/
  exact RingHom.ext eq
  /-
    🎉 no goals
  -/


/-- Extensionality of the `IsAdjoinRoot` structure itself. See `IsAdjoinRootMonic.ext_elem`
for extensionality of the ring elements. -/
@[ext]
theorem ext (h h' : IsAdjoinRoot S f) (eq : h.root = h'.root) : h = h' :=
                           /-
                             R : Type u
                             S : Type v
                             inst✝² : CommRing R
                             inst✝¹ : Ring S
                             f : Polynomial R
                             inst✝ : Algebra R S
                             h h' : IsAdjoinRoot S f
                             eq : Eq h.root h'.root
                             x : Polynomial R
                             ⊢ Eq (h.map x) (h'.map x)
                           -/
  h.ext_map h' fun x => by rw [← h.aeval_eq, ← h'.aeval_eq, eq]
                           /-
                             🎉 no goals
                           -/


/-- Auxiliary lemma for `IsAdjoinRoot.lift` -/
theorem eval₂_repr_eq_eval₂_of_map_eq (h : IsAdjoinRoot S f) (z : S) (w : R[X])
    (hzw : h.map w = z) : (h.repr z).eval₂ i x = w.eval₂ i x := by
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    f : Polynomial R
    inst✝¹ : Algebra R S
    T : Type u_1
    inst✝ : CommRing T
    i : RingHom R T
    x : T
    hx : Eq (Polynomial.eval₂ i x f) 0
    h : IsAdjoinRoot S f
    z : S
    w : Polynomial R
    hzw : Eq (h.map w) z
    ⊢ Eq (Polynomial.eval₂ i x (h.repr z)) (Polynomial.eval₂ i x w)
  -/
  rw [eq_comm, ← sub_eq_zero, ← h.map_repr z, ← map_sub, h.map_eq_zero_iff] at hzw
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    f : Polynomial R
    inst✝¹ : Algebra R S
    T : Type u_1
    inst✝ : CommRing T
    i : RingHom R T
    x : T
    hx : Eq (Polynomial.eval₂ i x f) 0
    h : IsAdjoinRoot S f
    z : S
    w : Polynomial R
    hzw : Dvd.dvd f (HSub.hSub (h.repr z) w)
    ⊢ Eq (Polynomial.eval₂ i x (h.repr z)) (Polynomial.eval₂ i x w)
  -/
  obtain ⟨y, hy⟩ := hzw
  /-
    case intro
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    f : Polynomial R
    inst✝¹ : Algebra R S
    T : Type u_1
    inst✝ : CommRing T
    i : RingHom R T
    x : T
    hx : Eq (Polynomial.eval₂ i x f) 0
    h : IsAdjoinRoot S f
    z : S
    w y : Polynomial R
    hy : Eq (HSub.hSub (h.repr z) w) (HMul.hMul f y)
    ⊢ Eq (Polynomial.eval₂ i x (h.repr z)) (Polynomial.eval₂ i x w)
  -/
  rw [← sub_eq_zero, ← eval₂_sub, hy, eval₂_mul, hx, zero_mul]
  /-
    🎉 no goals
  -/


/-- Lift a ring homomorphism `R →+* T` to `S →+* T` by specifying a root `x` of `f` in `T`,
where `S` is given by adjoining a root of `f` to `R`. -/
def lift (h : IsAdjoinRoot S f) (hx : f.eval₂ i x = 0) : S →+* T where
  toFun z := (h.repr z).eval₂ i x
  map_zero' := by
    /-
      R : Type u
      S : Type v
      inst✝³ : CommRing R
      inst✝² : Ring S
      f : Polynomial R
      inst✝¹ : Algebra R S
      T : Type u_1
      inst✝ : CommRing T
      i : RingHom R T
      x : T
      hx✝ : Eq (Polynomial.eval₂ i x f) 0
      h : IsAdjoinRoot S f
      hx : Eq (Polynomial.eval₂ i x f) 0
      ⊢ Eq ((↑{ toFun := fun z => Polynomial.eval₂ i x (h.repr z), map_one' := ⋯, ma …
    -/
    dsimp only -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10752): added `dsimp only`
    /-
      R : Type u
      S : Type v
      inst✝³ : CommRing R
      inst✝² : Ring S
      f : Polynomial R
      inst✝¹ : Algebra R S
      T : Type u_1
      inst✝ : CommRing T
      i : RingHom R T
      x : T
      hx✝ : Eq (Polynomial.eval₂ i x f) 0
      h : IsAdjoinRoot S f
      hx : Eq (Polynomial.eval₂ i x f) 0
      ⊢ Eq (Polynomial.eval₂ i x (h.repr 0)) 0
    -/
    rw [h.eval₂_repr_eq_eval₂_of_map_eq hx _ _ (map_zero _), eval₂_zero]
    /-
      🎉 no goals
    -/
  map_add' z w := by
    /-
      R : Type u
      S : Type v
      inst✝³ : CommRing R
      inst✝² : Ring S
      f : Polynomial R
      inst✝¹ : Algebra R S
      T : Type u_1
      inst✝ : CommRing T
      i : RingHom R T
      x : T
      hx✝ : Eq (Polynomial.eval₂ i x f) 0
      h : IsAdjoinRoot S f
      hx : Eq (Polynomial.eval₂ i x f) 0
      z w : S
      ⊢ Eq ((↑{ toFun := fun z => Polynomial.eval₂ i x (h.repr z), map_one' := ⋯, ma …
    -/
    /-
      R : Type u
      S : Type v
      inst✝³ : CommRing R
      inst✝² : Ring S
      f : Polynomial R
      inst✝¹ : Algebra R S
      T : Type u_1
      inst✝ : CommRing T
      i : RingHom R T
      x : T
      hx✝ : Eq (Polynomial.eval₂ i x f) 0
      h : IsAdjoinRoot S f
      hx : Eq (Polynomial.eval₂ i x f) 0
      ⊢ Eq ((fun z => Polynomial.eval₂ i x (h.repr z)) 1) 1
    -/
    dsimp only -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10752): added `dsimp only`
    /-
      R : Type u
      S : Type v
      inst✝³ : CommRing R
      inst✝² : Ring S
      f : Polynomial R
      inst✝¹ : Algebra R S
      T : Type u_1
      inst✝ : CommRing T
      i : RingHom R T
      x : T
      hx✝ : Eq (Polynomial.eval₂ i x f) 0
      h : IsAdjoinRoot S f
      hx : Eq (Polynomial.eval₂ i x f) 0
      ⊢ Eq (Polynomial.eval₂ i x (h.repr 1)) 1
    -/
    /-
      R : Type u
      S : Type v
      inst✝³ : CommRing R
      inst✝² : Ring S
      f : Polynomial R
      inst✝¹ : Algebra R S
      T : Type u_1
      inst✝ : CommRing T
      i : RingHom R T
      x : T
      hx✝ : Eq (Polynomial.eval₂ i x f) 0
      h : IsAdjoinRoot S f
      hx : Eq (Polynomial.eval₂ i x f) 0
      z w : S
      ⊢ Eq (Polynomial.eval₂ i x (h.repr (HAdd.hAdd z w))) (HAdd.hAdd (Polynomial.ev …
    -/
    /-
      🎉 no goals
    -/
    rw [h.eval₂_repr_eq_eval₂_of_map_eq hx _ (h.repr z + h.repr w), eval₂_add]
    /-
      R : Type u
      S : Type v
      inst✝³ : CommRing R
      inst✝² : Ring S
      f : Polynomial R
      inst✝¹ : Algebra R S
      T : Type u_1
      inst✝ : CommRing T
      i : RingHom R T
      x : T
      hx✝ : Eq (Polynomial.eval₂ i x f) 0
      h : IsAdjoinRoot S f
      hx : Eq (Polynomial.eval₂ i x f) 0
      z w : S
      ⊢ Eq ({ toFun := fun z => Polynomial.eval₂ i x (h.repr z), map_one' := ⋯ }.toF …
    -/
    /-
      R : Type u
      S : Type v
      inst✝³ : CommRing R
      inst✝² : Ring S
      f : Polynomial R
      inst✝¹ : Algebra R S
      T : Type u_1
      inst✝ : CommRing T
      i : RingHom R T
      x : T
      hx✝ : Eq (Polynomial.eval₂ i x f) 0
      h : IsAdjoinRoot S f
      hx : Eq (Polynomial.eval₂ i x f) 0
      z w : S
      ⊢ Eq (h.map (HAdd.hAdd (h.repr z) (h.repr w))) (HAdd.hAdd z w)
    -/
    /-
      R : Type u
      S : Type v
      inst✝³ : CommRing R
      inst✝² : Ring S
      f : Polynomial R
      inst✝¹ : Algebra R S
      T : Type u_1
      inst✝ : CommRing T
      i : RingHom R T
      x : T
      hx✝ : Eq (Polynomial.eval₂ i x f) 0
      h : IsAdjoinRoot S f
      hx : Eq (Polynomial.eval₂ i x f) 0
      z w : S
      ⊢ Eq (Polynomial.eval₂ i x (h.repr (HMul.hMul z w))) (HMul.hMul (Polynomial.ev …
    -/
    rw [map_add, map_repr, map_repr]
    /-
      R : Type u
      S : Type v
      inst✝³ : CommRing R
      inst✝² : Ring S
      f : Polynomial R
      inst✝¹ : Algebra R S
      T : Type u_1
      inst✝ : CommRing T
      i : RingHom R T
      x : T
      hx✝ : Eq (Polynomial.eval₂ i x f) 0
      h : IsAdjoinRoot S f
      hx : Eq (Polynomial.eval₂ i x f) 0
      z w : S
      ⊢ Eq (h.map (HMul.hMul (h.repr z) (h.repr w))) (HMul.hMul z w)
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
  map_one' := by
    beta_reduce -- Porting note (https://github.com/leanprover-community/mathlib4/issues/12129): additional beta reduction needed
    rw [h.eval₂_repr_eq_eval₂_of_map_eq hx _ _ (map_one _), eval₂_one]
  map_mul' z w := by
    dsimp only -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10752): added `dsimp only`
    rw [h.eval₂_repr_eq_eval₂_of_map_eq hx _ (h.repr z * h.repr w), eval₂_mul]
    rw [map_mul, map_repr, map_repr]


@[simp]
theorem lift_map (h : IsAdjoinRoot S f) (z : R[X]) : h.lift i x hx (h.map z) = z.eval₂ i x := by
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    f : Polynomial R
    inst✝¹ : Algebra R S
    T : Type u_1
    inst✝ : CommRing T
    i : RingHom R T
    x : T
    hx : Eq (Polynomial.eval₂ i x f) 0
    h : IsAdjoinRoot S f
    z : Polynomial R
    ⊢ Eq ((IsAdjoinRoot.lift i x h hx) (h.map z)) (Polynomial.eval₂ i x z)
  -/
  rw [lift, RingHom.coe_mk]
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    f : Polynomial R
    inst✝¹ : Algebra R S
    T : Type u_1
    inst✝ : CommRing T
    i : RingHom R T
    x : T
    hx : Eq (Polynomial.eval₂ i x f) 0
    h : IsAdjoinRoot S f
    z : Polynomial R
    ⊢ Eq ({ toFun := fun z => Polynomial.eval₂ i x (h.repr z), map_one' := ⋯, map_ …
  -/
  dsimp -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11227):added a `dsimp`
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    f : Polynomial R
    inst✝¹ : Algebra R S
    T : Type u_1
    inst✝ : CommRing T
    i : RingHom R T
    x : T
    hx : Eq (Polynomial.eval₂ i x f) 0
    h : IsAdjoinRoot S f
    z : Polynomial R
    ⊢ Eq (Polynomial.eval₂ i x (h.repr (h.map z))) (Polynomial.eval₂ i x z)
  -/
  rw [h.eval₂_repr_eq_eval₂_of_map_eq hx _ _ rfl]
  /-
    🎉 no goals
  -/


@[simp]
theorem lift_root (h : IsAdjoinRoot S f) : h.lift i x hx h.root = x := by
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    f : Polynomial R
    inst✝¹ : Algebra R S
    T : Type u_1
    inst✝ : CommRing T
    i : RingHom R T
    x : T
    hx : Eq (Polynomial.eval₂ i x f) 0
    h : IsAdjoinRoot S f
    ⊢ Eq ((IsAdjoinRoot.lift i x h hx) h.root) x
  -/
  rw [← h.map_X, lift_map, eval₂_X]
  /-
    🎉 no goals
  -/


@[simp]
theorem lift_algebraMap (h : IsAdjoinRoot S f) (a : R) :
                                                 /-
                                                   R : Type u
                                                   S : Type v
                                                   inst✝³ : CommRing R
                                                   inst✝² : Ring S
                                                   f : Polynomial R
                                                   inst✝¹ : Algebra R S
                                                   T : Type u_1
                                                   inst✝ : CommRing T
                                                   i : RingHom R T
                                                   x : T
                                                   hx : Eq (Polynomial.eval₂ i x f) 0
                                                   h : IsAdjoinRoot S f
                                                   a : R
                                                   ⊢ Eq ((IsAdjoinRoot.lift i x h hx) ((algebraMap R S) a)) (i a)
                                                 -/
    h.lift i x hx (algebraMap R S a) = i a := by rw [h.algebraMap_apply, lift_map, eval₂_C]
                                                 /-
                                                   🎉 no goals
                                                 -/


/-- Auxiliary lemma for `apply_eq_lift` -/
theorem apply_eq_lift (h : IsAdjoinRoot S f) (g : S →+* T) (hmap : ∀ a, g (algebraMap R S a) = i a)
    (hroot : g h.root = x) (a : S) : g a = h.lift i x hx a := by
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    f : Polynomial R
    inst✝¹ : Algebra R S
    T : Type u_1
    inst✝ : CommRing T
    i : RingHom R T
    x : T
    hx : Eq (Polynomial.eval₂ i x f) 0
    h : IsAdjoinRoot S f
    g : RingHom S T
    hmap : ∀ (a : R), Eq (g ((algebraMap R S) a)) (i a)
    hroot : Eq (g h.root) x
    a : S
    ⊢ Eq (g a) ((IsAdjoinRoot.lift i x h hx) a)
  -/
  rw [← h.map_repr a, Polynomial.as_sum_range_C_mul_X_pow (h.repr a)]
  simp only [map_sum, map_mul, map_pow, h.map_X, hroot, ← h.algebraMap_apply, hmap, lift_root,
    lift_algebraMap]


/-- Unicity of `lift`: a map that agrees on `R` and `h.root` agrees with `lift` everywhere. -/
theorem eq_lift (h : IsAdjoinRoot S f) (g : S →+* T) (hmap : ∀ a, g (algebraMap R S a) = i a)
    (hroot : g h.root = x) : g = h.lift i x hx :=
  RingHom.ext (h.apply_eq_lift hx g hmap hroot)


/-- Lift the algebra map `R → T` to `S →ₐ[R] T` by specifying a root `x` of `f` in `T`,
where `S` is given by adjoining a root of `f` to `R`. -/
def liftHom (h : IsAdjoinRoot S f) : S →ₐ[R] T :=
  { h.lift (algebraMap R T) x hx' with commutes' := fun a => h.lift_algebraMap hx' a }


@[simp]
theorem coe_liftHom (h : IsAdjoinRoot S f) :
    (h.liftHom x hx' : S →+* T) = h.lift (algebraMap R T) x hx' := rfl


theorem lift_algebraMap_apply (h : IsAdjoinRoot S f) (z : S) :
    h.lift (algebraMap R T) x hx' z = h.liftHom x hx' z := rfl


@[simp]
theorem liftHom_map (h : IsAdjoinRoot S f) (z : R[X]) : h.liftHom x hx' (h.map z) = aeval x z := by
  /-
    R : Type u
    S : Type v
    inst✝⁴ : CommRing R
    inst✝³ : Ring S
    f : Polynomial R
    inst✝² : Algebra R S
    T : Type u_1
    inst✝¹ : CommRing T
    x : T
    inst✝ : Algebra R T
    hx' : Eq ((Polynomial.aeval x) f) 0
    h : IsAdjoinRoot S f
    z : Polynomial R
    ⊢ Eq ((IsAdjoinRoot.liftHom x hx' h) (h.map z)) ((Polynomial.aeval x) z)
  -/
  rw [← lift_algebraMap_apply, lift_map, aeval_def]
  /-
    🎉 no goals
  -/


@[simp]
theorem liftHom_root (h : IsAdjoinRoot S f) : h.liftHom x hx' h.root = x := by
  /-
    R : Type u
    S : Type v
    inst✝⁴ : CommRing R
    inst✝³ : Ring S
    f : Polynomial R
    inst✝² : Algebra R S
    T : Type u_1
    inst✝¹ : CommRing T
    x : T
    inst✝ : Algebra R T
    hx' : Eq ((Polynomial.aeval x) f) 0
    h : IsAdjoinRoot S f
    ⊢ Eq ((IsAdjoinRoot.liftHom x hx' h) h.root) x
  -/
  rw [← lift_algebraMap_apply, lift_root]
  /-
    🎉 no goals
  -/


/-- Unicity of `liftHom`: a map that agrees on `h.root` agrees with `liftHom` everywhere. -/
theorem eq_liftHom (h : IsAdjoinRoot S f) (g : S →ₐ[R] T) (hroot : g h.root = x) :
    g = h.liftHom x hx' :=
  AlgHom.ext (h.apply_eq_lift hx' g g.commutes hroot)


/-- `AdjoinRoot f` is indeed given by adjoining a root of `f`. -/
protected def isAdjoinRoot : IsAdjoinRoot (AdjoinRoot f) f where
  map := AdjoinRoot.mk f
  map_surjective := Ideal.Quotient.mk_surjective
  ker_map := by
    /-
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : Ring S
      f : Polynomial R
      inst✝ : Algebra R S
      ⊢ Eq (RingHom.ker (AdjoinRoot.mk f)) (Ideal.span (Singleton.singleton f))
    -/
    ext
    rw [RingHom.mem_ker, ← @AdjoinRoot.mk_self _ _ f, AdjoinRoot.mk_eq_mk, Ideal.mem_span_singleton,
      ← dvd_add_left (dvd_refl f), sub_add_cancel]
  algebraMap_eq := AdjoinRoot.algebraMap_eq f


/-- `AdjoinRoot f` is indeed given by adjoining a root of `f`. If `f` is monic this is more
powerful than `AdjoinRoot.isAdjoinRoot`. -/
protected def isAdjoinRootMonic (hf : Monic f) : IsAdjoinRootMonic (AdjoinRoot f) f :=
  { AdjoinRoot.isAdjoinRoot f with Monic := hf }


@[simp]
theorem isAdjoinRoot_map_eq_mk : (AdjoinRoot.isAdjoinRoot f).map = AdjoinRoot.mk f :=
  rfl


@[simp]
theorem isAdjoinRootMonic_map_eq_mk (hf : f.Monic) :
    (AdjoinRoot.isAdjoinRootMonic f hf).map = AdjoinRoot.mk f :=
  rfl


@[simp]
theorem isAdjoinRoot_root_eq_root : (AdjoinRoot.isAdjoinRoot f).root = AdjoinRoot.root f := by
  /-
    R : Type u
    inst✝ : CommRing R
    f : Polynomial R
    ⊢ Eq (AdjoinRoot.isAdjoinRoot f).root (AdjoinRoot.root f)
  -/
  simp only [IsAdjoinRoot.root, AdjoinRoot.root, AdjoinRoot.isAdjoinRoot_map_eq_mk]
  /-
    🎉 no goals
  -/


@[simp]
theorem isAdjoinRootMonic_root_eq_root (hf : Monic f) :
    (AdjoinRoot.isAdjoinRootMonic f hf).root = AdjoinRoot.root f := by
  /-
    R : Type u
    inst✝ : CommRing R
    f : Polynomial R
    hf : f.Monic
    ⊢ Eq (AdjoinRoot.isAdjoinRootMonic f hf).root (AdjoinRoot.root f)
  -/
  simp only [IsAdjoinRoot.root, AdjoinRoot.root, AdjoinRoot.isAdjoinRootMonic_map_eq_mk]
  /-
    🎉 no goals
  -/


theorem map_modByMonic (h : IsAdjoinRootMonic S f) (g : R[X]) : h.map (g %ₘ f) = h.map g := by
  rw [← RingHom.sub_mem_ker_iff, mem_ker_map, modByMonic_eq_sub_mul_div _ h.Monic, sub_right_comm,
    sub_self, zero_sub, dvd_neg]
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    f : Polynomial R
    inst✝ : Algebra R S
    h : IsAdjoinRootMonic S f
    g : Polynomial R
    ⊢ Dvd.dvd f (HMul.hMul f (g.divByMonic f))
  -/
  exact ⟨_, rfl⟩
  /-
    🎉 no goals
  -/


theorem modByMonic_repr_map (h : IsAdjoinRootMonic S f) (g : R[X]) :
    h.repr (h.map g) %ₘ f = g %ₘ f :=
                                         /-
                                           R : Type u
                                           S : Type v
                                           inst✝² : CommRing R
                                           inst✝¹ : Ring S
                                           f : Polynomial R
                                           inst✝ : Algebra R S
                                           h : IsAdjoinRootMonic S f
                                           g : Polynomial R
                                           ⊢ Dvd.dvd f (HSub.hSub (h.repr (h.map g)) g)
                                         -/
  modByMonic_eq_of_dvd_sub h.Monic <| by rw [← h.mem_ker_map, RingHom.sub_mem_ker_iff, map_repr]
                                         /-
                                           🎉 no goals
                                         -/


/-- `IsAdjoinRoot.modByMonicHom` sends the equivalence class of `f` mod `g` to `f %ₘ g`. -/
def modByMonicHom (h : IsAdjoinRootMonic S f) : S →ₗ[R] R[X] where
  toFun x := h.repr x %ₘ f
  map_add' x y := by
    conv_lhs =>
      rw [← h.map_repr x, ← h.map_repr y, ← map_add]
      beta_reduce -- Porting note (https://github.com/leanprover-community/mathlib4/issues/12129): additional beta reduction needed
      rw [h.modByMonic_repr_map, add_modByMonic]
  map_smul' c x := by
    /-
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : Ring S
      f : Polynomial R
      inst✝ : Algebra R S
      h : IsAdjoinRootMonic S f
      c : R
      x : S
      ⊢ Eq ({ toFun := fun x => (h.repr x).modByMonic f, map_add' := ⋯ }.toFun (HSMu …
    -/
    rw [RingHom.id_apply, ← h.map_repr x, Algebra.smul_def, h.algebraMap_apply, ← map_mul]
    /-
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : Ring S
      f : Polynomial R
      inst✝ : Algebra R S
      h : IsAdjoinRootMonic S f
      c : R
      x : S
      ⊢ Eq ({ toFun := fun x => (h.repr x).modByMonic f, map_add' := ⋯ }.toFun (h.ma …
    -/
    dsimp only -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10752): added `dsimp only`
    /-
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : Ring S
      f : Polynomial R
      inst✝ : Algebra R S
      h : IsAdjoinRootMonic S f
      c : R
      x : S
      ⊢ Eq ((h.repr (h.map (HMul.hMul (Polynomial.C c) (h.repr x)))).modByMonic f) ( …
    -/
    rw [h.modByMonic_repr_map, ← smul_eq_C_mul, smul_modByMonic, h.map_repr]
    /-
      🎉 no goals
    -/


@[simp]
theorem modByMonicHom_map (h : IsAdjoinRootMonic S f) (g : R[X]) :
    h.modByMonicHom (h.map g) = g %ₘ f := h.modByMonic_repr_map g


@[simp]
theorem map_modByMonicHom (h : IsAdjoinRootMonic S f) (x : S) : h.map (h.modByMonicHom x) = x := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    f : Polynomial R
    inst✝ : Algebra R S
    h : IsAdjoinRootMonic S f
    x : S
    ⊢ Eq (h.map (h.modByMonicHom x)) x
  -/
  rw [modByMonicHom, LinearMap.coe_mk]
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    f : Polynomial R
    inst✝ : Algebra R S
    h : IsAdjoinRootMonic S f
    x : S
    ⊢ Eq (h.map ({ toFun := fun x => (h.repr x).modByMonic f, map_add' := ⋯ } x)) x
  -/
  dsimp -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11227):added a `dsimp`
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    f : Polynomial R
    inst✝ : Algebra R S
    h : IsAdjoinRootMonic S f
    x : S
    ⊢ Eq (h.map ((h.repr x).modByMonic f)) x
  -/
  rw [map_modByMonic, map_repr]
  /-
    🎉 no goals
  -/


@[simp]
theorem modByMonicHom_root_pow (h : IsAdjoinRootMonic S f) {n : ℕ} (hdeg : n < natDegree f) :
    h.modByMonicHom (h.root ^ n) = X ^ n := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    f : Polynomial R
    inst✝ : Algebra R S
    h : IsAdjoinRootMonic S f
    n : Nat
    hdeg : LT.lt n f.natDegree
    ⊢ Eq (h.modByMonicHom (HPow.hPow h.root n)) (HPow.hPow Polynomial.X n)
  -/
  nontriviality R
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    f : Polynomial R
    inst✝ : Algebra R S
    h : IsAdjoinRootMonic S f
    n : Nat
    hdeg : LT.lt n f.natDegree
    a✝ : Nontrivial R
    ⊢ Eq (h.modByMonicHom (HPow.hPow h.root n)) (HPow.hPow Polynomial.X n)
  -/
  rw [← h.map_X, ← map_pow, modByMonicHom_map, modByMonic_eq_self_iff h.Monic, degree_X_pow]
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    f : Polynomial R
    inst✝ : Algebra R S
    h : IsAdjoinRootMonic S f
    n : Nat
    hdeg : LT.lt n f.natDegree
    a✝ : Nontrivial R
    ⊢ LT.lt (↑n) f.degree
  -/
  contrapose! hdeg
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    f : Polynomial R
    inst✝ : Algebra R S
    h : IsAdjoinRootMonic S f
    n : Nat
    a✝ : Nontrivial R
    hdeg : LE.le f.degree ↑n
    ⊢ LE.le f.natDegree n
  -/
  simpa [natDegree_le_iff_degree_le] using hdeg
  /-
    🎉 no goals
  -/


@[simp]
theorem modByMonicHom_root (h : IsAdjoinRootMonic S f) (hdeg : 1 < natDegree f) :
                                     /-
                                       R : Type u
                                       S : Type v
                                       inst✝² : CommRing R
                                       inst✝¹ : Ring S
                                       f : Polynomial R
                                       inst✝ : Algebra R S
                                       h : IsAdjoinRootMonic S f
                                       hdeg : LT.lt 1 f.natDegree
                                       ⊢ Eq (h.modByMonicHom h.root) Polynomial.X
                                     -/
    h.modByMonicHom h.root = X := by simpa using modByMonicHom_root_pow h hdeg
                                     /-
                                       🎉 no goals
                                     -/


/-- The basis on `S` generated by powers of `h.root`.

Auxiliary definition for `IsAdjoinRootMonic.powerBasis`. -/
def basis (h : IsAdjoinRootMonic S f) : Basis (Fin (natDegree f)) R S :=
  Basis.ofRepr
    { toFun := fun x => (h.modByMonicHom x).toFinsupp.comapDomain _ Fin.val_injective.injOn
      invFun := fun g => h.map (ofFinsupp (g.mapDomain Fin.val))
      left_inv := fun x => by
        /-
          R : Type u
          S : Type v
          inst✝² : CommRing R
          inst✝¹ : Ring S
          f : Polynomial R
          inst✝ : Algebra R S
          h : IsAdjoinRootMonic S f
          x : S
          ⊢ Eq ((fun g => h.map { toFinsupp := Finsupp.mapDomain Fin.val g }) ({ toFun : …
        -/
        cases subsingleton_or_nontrivial R
          /-
            case inl
            R : Type u
            S : Type v
            inst✝² : CommRing R
            inst✝¹ : Ring S
            f : Polynomial R
            inst✝ : Algebra R S
            h : IsAdjoinRootMonic S f
            x : S
            h✝ : Subsingleton R
            ⊢ Eq ((fun g => h.map { toFinsupp := Finsupp.mapDomain Fin.val g }) ({ toFun : …
          -/
        · subsingleton [h.subsingleton]
          /-
            🎉 no goals
          -/
        /-
          case inr
          R : Type u
          S : Type v
          inst✝² : CommRing R
          inst✝¹ : Ring S
          f : Polynomial R
          inst✝ : Algebra R S
          h : IsAdjoinRootMonic S f
          x : S
          h✝ : Nontrivial R
          ⊢ Eq ((fun g => h.map { toFinsupp := Finsupp.mapDomain Fin.val g }) ({ toFun : …
        -/
        simp only
        /-
          case inr
          R : Type u
          S : Type v
          inst✝² : CommRing R
          inst✝¹ : Ring S
          f : Polynomial R
          inst✝ : Algebra R S
          h : IsAdjoinRootMonic S f
          x : S
          h✝ : Nontrivial R
          ⊢ Eq (h.map { toFinsupp := Finsupp.mapDomain Fin.val (Finsupp.comapDomain Fin. …
        -/
        rw [Finsupp.mapDomain_comapDomain, Polynomial.eta, h.map_modByMonicHom x]
          /-
            case inr.hf
            R : Type u
            S : Type v
            inst✝² : CommRing R
            inst✝¹ : Ring S
            f : Polynomial R
            inst✝ : Algebra R S
            h : IsAdjoinRootMonic S f
            x : S
            h✝ : Nontrivial R
            ⊢ Function.Injective Fin.val
          -/
        · exact Fin.val_injective
          /-
            🎉 no goals
          -/
        /-
          case inr.hl
          R : Type u
          S : Type v
          inst✝² : CommRing R
          inst✝¹ : Ring S
          f : Polynomial R
          inst✝ : Algebra R S
          h : IsAdjoinRootMonic S f
          x : S
          h✝ : Nontrivial R
          ⊢ HasSubset.Subset (↑(h.modByMonicHom x).toFinsupp.support) (Set.range Fin.val)
        -/
        intro i hi
        /-
          case inr.hl
          R : Type u
          S : Type v
          inst✝² : CommRing R
          inst✝¹ : Ring S
          f : Polynomial R
          inst✝ : Algebra R S
          h : IsAdjoinRootMonic S f
          x : S
          h✝ : Nontrivial R
          i : Nat
          hi : Membership.mem (↑(h.modByMonicHom x).toFinsupp.support) i
          ⊢ Membership.mem (Set.range Fin.val) i
        -/
        refine Set.mem_range.mpr ⟨⟨i, ?_⟩, rfl⟩
        /-
          case inr.hl
          R : Type u
          S : Type v
          inst✝² : CommRing R
          inst✝¹ : Ring S
          f : Polynomial R
          inst✝ : Algebra R S
          h : IsAdjoinRootMonic S f
          x : S
          h✝ : Nontrivial R
          i : Nat
          hi : Membership.mem (↑(h.modByMonicHom x).toFinsupp.support) i
          ⊢ LT.lt i f.natDegree
        -/
        contrapose! hi
        simp only [Polynomial.toFinsupp_apply, Classical.not_not, Finsupp.mem_support_iff, Ne,
          modByMonicHom, LinearMap.coe_mk, Finset.mem_coe]
        /-
          case inr.hl
          R : Type u
          S : Type v
          inst✝² : CommRing R
          inst✝¹ : Ring S
          f : Polynomial R
          inst✝ : Algebra R S
          h : IsAdjoinRootMonic S f
          x : S
          h✝ : Nontrivial R
          i : Nat
          hi : LE.le f.natDegree i
          ⊢ Eq (({ toFun := fun x => (h.repr x).modByMonic f, map_add' := ⋯ } x).coeff i …
        -/
        by_cases hx : h.toIsAdjoinRoot.repr x %ₘ f = 0
          /-
            case pos
            R : Type u
            S : Type v
            inst✝² : CommRing R
            inst✝¹ : Ring S
            f : Polynomial R
            inst✝ : Algebra R S
            h : IsAdjoinRootMonic S f
            x : S
            h✝ : Nontrivial R
            i : Nat
            hi : LE.le f.natDegree i
            hx : Eq ((h.repr x).modByMonic f) 0
            ⊢ Eq (({ toFun := fun x => (h.repr x).modByMonic f, map_add' := ⋯ } x).coeff i …
          -/
        · simp [hx]
          /-
            🎉 no goals
          -/
        /-
          case neg
          R : Type u
          S : Type v
          inst✝² : CommRing R
          inst✝¹ : Ring S
          f : Polynomial R
          inst✝ : Algebra R S
          h : IsAdjoinRootMonic S f
          x : S
          h✝ : Nontrivial R
          i : Nat
          hi : LE.le f.natDegree i
          hx : Not (Eq ((h.repr x).modByMonic f) 0)
          ⊢ Eq (({ toFun := fun x => (h.repr x).modByMonic f, map_add' := ⋯ } x).coeff i …
        -/
        refine coeff_eq_zero_of_natDegree_lt (lt_of_lt_of_le ?_ hi)
        /-
          case neg
          R : Type u
          S : Type v
          inst✝² : CommRing R
          inst✝¹ : Ring S
          f : Polynomial R
          inst✝ : Algebra R S
          h : IsAdjoinRootMonic S f
          x : S
          h✝ : Nontrivial R
          i : Nat
          hi : LE.le f.natDegree i
          hx : Not (Eq ((h.repr x).modByMonic f) 0)
          ⊢ LT.lt ({ toFun := fun x => (h.repr x).modByMonic f, map_add' := ⋯ } x).natDe …
        -/
        dsimp -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11227):added a `dsimp`
        /-
          case neg
          R : Type u
          S : Type v
          inst✝² : CommRing R
          inst✝¹ : Ring S
          f : Polynomial R
          inst✝ : Algebra R S
          h : IsAdjoinRootMonic S f
          x : S
          h✝ : Nontrivial R
          i : Nat
          hi : LE.le f.natDegree i
          hx : Not (Eq ((h.repr x).modByMonic f) 0)
          ⊢ LT.lt ((h.repr x).modByMonic f).natDegree f.natDegree
        -/
        /-
          R : Type u
          S : Type v
          inst✝² : CommRing R
          inst✝¹ : Ring S
          f : Polynomial R
          inst✝ : Algebra R S
          h : IsAdjoinRootMonic S f
          x y : S
          ⊢ Eq ((fun x => Finsupp.comapDomain Fin.val (h.modByMonicHom x).toFinsupp ⋯) ( …
        -/
        rw [natDegree_lt_natDegree_iff hx]
        /-
          R : Type u
          S : Type v
          inst✝² : CommRing R
          inst✝¹ : Ring S
          f : Polynomial R
          inst✝ : Algebra R S
          h : IsAdjoinRootMonic S f
          x y : S
          ⊢ Eq (Finsupp.comapDomain Fin.val (h.modByMonicHom (HAdd.hAdd x y)).toFinsupp  …
        -/
        /-
          case neg
          R : Type u
          S : Type v
          inst✝² : CommRing R
          inst✝¹ : Ring S
          f : Polynomial R
          inst✝ : Algebra R S
          h : IsAdjoinRootMonic S f
          x : S
          h✝ : Nontrivial R
          i : Nat
          hi : LE.le f.natDegree i
          hx : Not (Eq ((h.repr x).modByMonic f) 0)
          ⊢ LT.lt ((h.repr x).modByMonic f).degree f.degree
        -/
        /-
          🎉 no goals
        -/
        exact degree_modByMonic_lt _ h.Monic
        /-
          🎉 no goals
        -/
      right_inv := fun g => by
        /-
          R : Type u
          S : Type v
          inst✝² : CommRing R
          inst✝¹ : Ring S
          f : Polynomial R
          inst✝ : Algebra R S
          h : IsAdjoinRootMonic S f
          g : Finsupp (Fin f.natDegree) R
          ⊢ Eq ({ toFun := fun x => Finsupp.comapDomain Fin.val (h.modByMonicHom x).toFi …
        -/
        /-
          R : Type u
          S : Type v
          inst✝² : CommRing R
          inst✝¹ : Ring S
          f : Polynomial R
          inst✝ : Algebra R S
          h : IsAdjoinRootMonic S f
          c : R
          x : S
          ⊢ Eq ({ toFun := fun x => Finsupp.comapDomain Fin.val (h.modByMonicHom x).toFi …
        -/
        nontriviality R
        /-
          R : Type u
          S : Type v
          inst✝² : CommRing R
          inst✝¹ : Ring S
          f : Polynomial R
          inst✝ : Algebra R S
          h : IsAdjoinRootMonic S f
          g : Finsupp (Fin f.natDegree) R
          a✝ : Nontrivial R
          ⊢ Eq ({ toFun := fun x => Finsupp.comapDomain Fin.val (h.modByMonicHom x).toFi …
        -/
        ext i
        /-
          case h
          R : Type u
          S : Type v
          inst✝² : CommRing R
          inst✝¹ : Ring S
          f : Polynomial R
          inst✝ : Algebra R S
          h : IsAdjoinRootMonic S f
          g : Finsupp (Fin f.natDegree) R
          a✝ : Nontrivial R
          i : Fin f.natDegree
          ⊢ Eq (({ toFun := fun x => Finsupp.comapDomain Fin.val (h.modByMonicHom x).toF …
        -/
        simp only [h.modByMonicHom_map, Finsupp.comapDomain_apply, Polynomial.toFinsupp_apply]
        /-
          case h
          R : Type u
          S : Type v
          inst✝² : CommRing R
          inst✝¹ : Ring S
          f : Polynomial R
          inst✝ : Algebra R S
          h : IsAdjoinRootMonic S f
          g : Finsupp (Fin f.natDegree) R
          a✝ : Nontrivial R
          i : Fin f.natDegree
          ⊢ Eq (({ toFinsupp := Finsupp.mapDomain Fin.val g }.modByMonic f).coeff ↑i) (g …
        -/
        rw [(Polynomial.modByMonic_eq_self_iff h.Monic).mpr, Polynomial.coeff]
          /-
            case h
            R : Type u
            S : Type v
            inst✝² : CommRing R
            inst✝¹ : Ring S
            f : Polynomial R
            inst✝ : Algebra R S
            h : IsAdjoinRootMonic S f
            g : Finsupp (Fin f.natDegree) R
            a✝ : Nontrivial R
            i : Fin f.natDegree
            ⊢ Eq ((Finsupp.mapDomain Fin.val g) ↑i) (g i)
          -/
        · rw [Finsupp.mapDomain_apply Fin.val_injective]
          /-
            🎉 no goals
          -/
        /-
          case h
          R : Type u
          S : Type v
          inst✝² : CommRing R
          inst✝¹ : Ring S
          f : Polynomial R
          inst✝ : Algebra R S
          h : IsAdjoinRootMonic S f
          g : Finsupp (Fin f.natDegree) R
          a✝ : Nontrivial R
          i : Fin f.natDegree
          ⊢ LT.lt { toFinsupp := Finsupp.mapDomain Fin.val g }.degree f.degree
        -/
        rw [degree_eq_natDegree h.Monic.ne_zero, degree_lt_iff_coeff_zero]
        /-
          case h
          R : Type u
          S : Type v
          inst✝² : CommRing R
          inst✝¹ : Ring S
          f : Polynomial R
          inst✝ : Algebra R S
          h : IsAdjoinRootMonic S f
          g : Finsupp (Fin f.natDegree) R
          a✝ : Nontrivial R
          i : Fin f.natDegree
          ⊢ ∀ (m : Nat), LE.le f.natDegree m → Eq ({ toFinsupp := Finsupp.mapDomain Fin. …
        -/
        intro m hm
        /-
          case h
          R : Type u
          S : Type v
          inst✝² : CommRing R
          inst✝¹ : Ring S
          f : Polynomial R
          inst✝ : Algebra R S
          h : IsAdjoinRootMonic S f
          g : Finsupp (Fin f.natDegree) R
          a✝ : Nontrivial R
          i : Fin f.natDegree
          m : Nat
          hm : LE.le f.natDegree m
          ⊢ Eq ({ toFinsupp := Finsupp.mapDomain Fin.val g }.coeff m) 0
        -/
        rw [Polynomial.coeff]
        /-
          case h
          R : Type u
          S : Type v
          inst✝² : CommRing R
          inst✝¹ : Ring S
          f : Polynomial R
          inst✝ : Algebra R S
          h : IsAdjoinRootMonic S f
          g : Finsupp (Fin f.natDegree) R
          a✝ : Nontrivial R
          i : Fin f.natDegree
          m : Nat
          hm : LE.le f.natDegree m
          ⊢ Eq ((Finsupp.mapDomain Fin.val g) m) 0
        -/
        rw [Finsupp.mapDomain_notin_range]
        /-
          case h.h
          R : Type u
          S : Type v
          inst✝² : CommRing R
          inst✝¹ : Ring S
          f : Polynomial R
          inst✝ : Algebra R S
          h : IsAdjoinRootMonic S f
          g : Finsupp (Fin f.natDegree) R
          a✝ : Nontrivial R
          i : Fin f.natDegree
          m : Nat
          hm : LE.le f.natDegree m
          ⊢ Not (Membership.mem (Set.range Fin.val) m)
        -/
        rw [Set.mem_range, not_exists]
        /-
          case h.h
          R : Type u
          S : Type v
          inst✝² : CommRing R
          inst✝¹ : Ring S
          f : Polynomial R
          inst✝ : Algebra R S
          h : IsAdjoinRootMonic S f
          g : Finsupp (Fin f.natDegree) R
          a✝ : Nontrivial R
          i : Fin f.natDegree
          m : Nat
          hm : LE.le f.natDegree m
          ⊢ ∀ (x : Fin f.natDegree), Not (Eq (↑x) m)
        -/
        rintro i rfl
        /-
          case h.h
          R : Type u
          S : Type v
          inst✝² : CommRing R
          inst✝¹ : Ring S
          f : Polynomial R
          inst✝ : Algebra R S
          h : IsAdjoinRootMonic S f
          g : Finsupp (Fin f.natDegree) R
          a✝ : Nontrivial R
          i✝ i : Fin f.natDegree
          hm : LE.le f.natDegree ↑i
          ⊢ False
        -/
        exact i.prop.not_le hm
        /-
          🎉 no goals
        -/
      map_add' := fun x y => by
        beta_reduce -- Porting note (https://github.com/leanprover-community/mathlib4/issues/12129): additional beta reduction needed
        rw [map_add, toFinsupp_add, Finsupp.comapDomain_add_of_injective Fin.val_injective]
      -- Porting note: the original simp proof with the same lemmas does not work
      -- See https://github.com/leanprover-community/mathlib4/issues/5026
      -- simp only [map_add, Finsupp.comapDomain_add_of_injective Fin.val_injective, toFinsupp_add]
      map_smul' := fun c x => by
        dsimp only -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10752): added `dsimp only`
        rw [map_smul, toFinsupp_smul, Finsupp.comapDomain_smul_of_injective Fin.val_injective,
          RingHom.id_apply] }
      -- Porting note: the original simp proof with the same lemmas does not work
      -- See https://github.com/leanprover-community/mathlib4/issues/5026
      -- simp only [map_smul, Finsupp.comapDomain_smul_of_injective Fin.val_injective,
      --   RingHom.id_apply, toFinsupp_smul] }


@[simp]
theorem basis_apply (h : IsAdjoinRootMonic S f) (i) : h.basis i = h.root ^ (i : ℕ) :=
  Basis.apply_eq_iff.mpr <|
    show (h.modByMonicHom (h.toIsAdjoinRoot.root ^ (i : ℕ))).toFinsupp.comapDomain _
          Fin.val_injective.injOn = Finsupp.single _ _ by
      /-
        R : Type u
        S : Type v
        inst✝² : CommRing R
        inst✝¹ : Ring S
        f : Polynomial R
        inst✝ : Algebra R S
        h : IsAdjoinRootMonic S f
        i : Fin f.natDegree
        ⊢ Eq (Finsupp.comapDomain Fin.val (h.modByMonicHom (HPow.hPow h.root ↑i)).toFi …
      -/
      ext j
      /-
        case h
        R : Type u
        S : Type v
        inst✝² : CommRing R
        inst✝¹ : Ring S
        f : Polynomial R
        inst✝ : Algebra R S
        h : IsAdjoinRootMonic S f
        i j : Fin f.natDegree
        ⊢ Eq ((Finsupp.comapDomain Fin.val (h.modByMonicHom (HPow.hPow h.root ↑i)).toF …
      -/
      rw [Finsupp.comapDomain_apply, modByMonicHom_root_pow]
        /-
          case h
          R : Type u
          S : Type v
          inst✝² : CommRing R
          inst✝¹ : Ring S
          f : Polynomial R
          inst✝ : Algebra R S
          h : IsAdjoinRootMonic S f
          i j : Fin f.natDegree
          ⊢ Eq ((HPow.hPow Polynomial.X ↑i).toFinsupp ↑j) ((Finsupp.single i 1) j)
        -/
      · rw [X_pow_eq_monomial, toFinsupp_monomial, Finsupp.single_apply_left Fin.val_injective]
        /-
          🎉 no goals
        -/
        /-
          case h.hdeg
          R : Type u
          S : Type v
          inst✝² : CommRing R
          inst✝¹ : Ring S
          f : Polynomial R
          inst✝ : Algebra R S
          h : IsAdjoinRootMonic S f
          i j : Fin f.natDegree
          ⊢ LT.lt (↑i) f.natDegree
        -/
      · exact i.is_lt
        /-
          🎉 no goals
        -/


theorem deg_pos [Nontrivial S] (h : IsAdjoinRootMonic S f) : 0 < natDegree f := by
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    f : Polynomial R
    inst✝¹ : Algebra R S
    inst✝ : Nontrivial S
    h : IsAdjoinRootMonic S f
    ⊢ LT.lt 0 f.natDegree
  -/
  rcases h.basis.index_nonempty with ⟨⟨i, hi⟩⟩
  /-
    case intro.mk
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    f : Polynomial R
    inst✝¹ : Algebra R S
    inst✝ : Nontrivial S
    h : IsAdjoinRootMonic S f
    i : Nat
    hi : LT.lt i f.natDegree
    ⊢ LT.lt 0 f.natDegree
  -/
  exact (Nat.zero_le _).trans_lt hi
  /-
    🎉 no goals
  -/


theorem deg_ne_zero [Nontrivial S] (h : IsAdjoinRootMonic S f) : natDegree f ≠ 0 :=
  h.deg_pos.ne'


/-- If `f` is monic, the powers of `h.root` form a basis. -/
@[simps! gen dim basis]
def powerBasis (h : IsAdjoinRootMonic S f) : PowerBasis R S where
  gen := h.root
  dim := natDegree f
  basis := h.basis
  basis_eq_pow := h.basis_apply


@[simp]
theorem basis_repr (h : IsAdjoinRootMonic S f) (x : S) (i : Fin (natDegree f)) :
    h.basis.repr x i = (h.modByMonicHom x).coeff (i : ℕ) := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    f : Polynomial R
    inst✝ : Algebra R S
    h : IsAdjoinRootMonic S f
    x : S
    i : Fin f.natDegree
    ⊢ Eq ((h.basis.repr x) i) ((h.modByMonicHom x).coeff ↑i)
  -/
  change (h.modByMonicHom x).toFinsupp.comapDomain _ Fin.val_injective.injOn i = _
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    f : Polynomial R
    inst✝ : Algebra R S
    h : IsAdjoinRootMonic S f
    x : S
    i : Fin f.natDegree
    ⊢ Eq ((Finsupp.comapDomain Fin.val (h.modByMonicHom x).toFinsupp ⋯) i) ((h.mod …
  -/
  rw [Finsupp.comapDomain_apply, Polynomial.toFinsupp_apply]
  /-
    🎉 no goals
  -/


theorem basis_one (h : IsAdjoinRootMonic S f) (hdeg : 1 < natDegree f) :
                                     /-
                                       R : Type u
                                       S : Type v
                                       inst✝² : CommRing R
                                       inst✝¹ : Ring S
                                       f : Polynomial R
                                       inst✝ : Algebra R S
                                       h : IsAdjoinRootMonic S f
                                       hdeg : LT.lt 1 f.natDegree
                                       ⊢ Eq (h.basis ⟨1, hdeg⟩) h.root
                                     -/
    h.basis ⟨1, hdeg⟩ = h.root := by rw [h.basis_apply, Fin.val_mk, pow_one]
                                     /-
                                       🎉 no goals
                                     -/


/-- `IsAdjoinRootMonic.liftPolyₗ` lifts a linear map on polynomials to a linear map on `S`. -/
@[simps!]
def liftPolyₗ {T : Type*} [AddCommGroup T] [Module R T] (h : IsAdjoinRootMonic S f)
    (g : R[X] →ₗ[R] T) : S →ₗ[R] T :=
  g.comp h.modByMonicHom


/-- `IsAdjoinRootMonic.coeff h x i` is the `i`th coefficient of the representative of `x : S`.
-/
def coeff (h : IsAdjoinRootMonic S f) : S →ₗ[R] ℕ → R :=
  h.liftPolyₗ
    { toFun := Polynomial.coeff
      map_add' := fun p q => funext (Polynomial.coeff_add p q)
      map_smul' := fun c p => funext (Polynomial.coeff_smul c p) }


theorem coeff_apply_lt (h : IsAdjoinRootMonic S f) (z : S) (i : ℕ) (hi : i < natDegree f) :
    h.coeff z i = h.basis.repr z ⟨i, hi⟩ := by
  simp only [coeff, LinearMap.comp_apply, Finsupp.lcoeFun_apply, Finsupp.lmapDomain_apply,
    LinearEquiv.coe_coe, liftPolyₗ_apply, LinearMap.coe_mk, h.basis_repr]
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    f : Polynomial R
    inst✝ : Algebra R S
    h : IsAdjoinRootMonic S f
    z : S
    i : Nat
    hi : LT.lt i f.natDegree
    ⊢ Eq ({ toFun := Polynomial.coeff, map_add' := ⋯ } (h.modByMonicHom z) i) ((h. …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem coeff_apply_coe (h : IsAdjoinRootMonic S f) (z : S) (i : Fin (natDegree f)) :
    h.coeff z i = h.basis.repr z i := h.coeff_apply_lt z i i.prop


theorem coeff_apply_le (h : IsAdjoinRootMonic S f) (z : S) (i : ℕ) (hi : natDegree f ≤ i) :
    h.coeff z i = 0 := by
  simp only [coeff, LinearMap.comp_apply, Finsupp.lcoeFun_apply, Finsupp.lmapDomain_apply,
    LinearEquiv.coe_coe, liftPolyₗ_apply, LinearMap.coe_mk, h.basis_repr]
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    f : Polynomial R
    inst✝ : Algebra R S
    h : IsAdjoinRootMonic S f
    z : S
    i : Nat
    hi : LE.le f.natDegree i
    ⊢ Eq ({ toFun := Polynomial.coeff, map_add' := ⋯ } (h.modByMonicHom z) i) 0
  -/
  nontriviality R
  exact
    Polynomial.coeff_eq_zero_of_degree_lt
      ((degree_modByMonic_lt _ h.Monic).trans_le (Polynomial.degree_le_of_natDegree_le hi))


theorem coeff_apply (h : IsAdjoinRootMonic S f) (z : S) (i : ℕ) :
    h.coeff z i = if hi : i < natDegree f then h.basis.repr z ⟨i, hi⟩ else 0 := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    f : Polynomial R
    inst✝ : Algebra R S
    h : IsAdjoinRootMonic S f
    z : S
    i : Nat
    ⊢ Eq (h.coeff z i) (dite (LT.lt i f.natDegree) (fun hi => (h.basis.repr z) ⟨i, …
  -/
  split_ifs with hi
    /-
      case pos
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : Ring S
      f : Polynomial R
      inst✝ : Algebra R S
      h : IsAdjoinRootMonic S f
      z : S
      i : Nat
      hi : LT.lt i f.natDegree
      ⊢ Eq (h.coeff z i) ((h.basis.repr z) ⟨i, hi⟩)
    -/
  · exact h.coeff_apply_lt z i hi
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : Ring S
      f : Polynomial R
      inst✝ : Algebra R S
      h : IsAdjoinRootMonic S f
      z : S
      i : Nat
      hi : Not (LT.lt i f.natDegree)
      ⊢ Eq (h.coeff z i) 0
    -/
  · exact h.coeff_apply_le z i (le_of_not_lt hi)
    /-
      🎉 no goals
    -/


theorem coeff_root_pow (h : IsAdjoinRootMonic S f) {n} (hn : n < natDegree f) :
    h.coeff (h.root ^ n) = Pi.single n 1 := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    f : Polynomial R
    inst✝ : Algebra R S
    h : IsAdjoinRootMonic S f
    n : Nat
    hn : LT.lt n f.natDegree
    ⊢ Eq (h.coeff (HPow.hPow h.root n)) (Pi.single n 1)
  -/
  ext i
  /-
    case h
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    f : Polynomial R
    inst✝ : Algebra R S
    h : IsAdjoinRootMonic S f
    n : Nat
    hn : LT.lt n f.natDegree
    i : Nat
    ⊢ Eq (h.coeff (HPow.hPow h.root n) i) (Pi.single n 1 i)
  -/
  rw [coeff_apply]
  /-
    case h
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    f : Polynomial R
    inst✝ : Algebra R S
    h : IsAdjoinRootMonic S f
    n : Nat
    hn : LT.lt n f.natDegree
    i : Nat
    ⊢ Eq (dite (LT.lt i f.natDegree) (fun hi => (h.basis.repr (HPow.hPow h.root n) …
  -/
  split_ifs with hi
  · calc
      h.basis.repr (h.root ^ n) ⟨i, _⟩ = h.basis.repr (h.basis ⟨n, hn⟩) ⟨i, hi⟩ := by
        rw [h.basis_apply, Fin.val_mk]
      _ = Pi.single (f := fun _ => R) ((⟨n, hn⟩ : Fin _) : ℕ) (1 : (fun _ => R) n)
        ↑(⟨i, _⟩ : Fin _) := by
        rw [h.basis.repr_self, ← Finsupp.single_eq_pi_single,
          Finsupp.single_apply_left Fin.val_injective]
      _ = Pi.single (f := fun _ => R) n 1 i := by rw [Fin.val_mk, Fin.val_mk]
    /-
      case neg
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : Ring S
      f : Polynomial R
      inst✝ : Algebra R S
      h : IsAdjoinRootMonic S f
      n : Nat
      hn : LT.lt n f.natDegree
      i : Nat
      hi : Not (LT.lt i f.natDegree)
      ⊢ Eq 0 (Pi.single n 1 i)
    -/
  · refine (Pi.single_eq_of_ne (f := fun _ => R) ?_ (1 : (fun _ => R) n)).symm
    /-
      case neg
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : Ring S
      f : Polynomial R
      inst✝ : Algebra R S
      h : IsAdjoinRootMonic S f
      n : Nat
      hn : LT.lt n f.natDegree
      i : Nat
      hi : Not (LT.lt i f.natDegree)
      ⊢ Ne i n
    -/
    rintro rfl
    /-
      case neg
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : Ring S
      f : Polynomial R
      inst✝ : Algebra R S
      h : IsAdjoinRootMonic S f
      i : Nat
      hi : Not (LT.lt i f.natDegree)
      hn : LT.lt i f.natDegree
      ⊢ False
    -/
    simp [hi] at hn
    /-
      🎉 no goals
    -/


theorem coeff_one [Nontrivial S] (h : IsAdjoinRootMonic S f) : h.coeff 1 = Pi.single 0 1 := by
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    f : Polynomial R
    inst✝¹ : Algebra R S
    inst✝ : Nontrivial S
    h : IsAdjoinRootMonic S f
    ⊢ Eq (h.coeff 1) (Pi.single 0 1)
  -/
  rw [← h.coeff_root_pow h.deg_pos, pow_zero]
  /-
    🎉 no goals
  -/


theorem coeff_root (h : IsAdjoinRootMonic S f) (hdeg : 1 < natDegree f) :
                                         /-
                                           R : Type u
                                           S : Type v
                                           inst✝² : CommRing R
                                           inst✝¹ : Ring S
                                           f : Polynomial R
                                           inst✝ : Algebra R S
                                           h : IsAdjoinRootMonic S f
                                           hdeg : LT.lt 1 f.natDegree
                                           ⊢ Eq (h.coeff h.root) (Pi.single 1 1)
                                         -/
    h.coeff h.root = Pi.single 1 1 := by rw [← h.coeff_root_pow hdeg, pow_one]
                                         /-
                                           🎉 no goals
                                         -/


theorem coeff_algebraMap [Nontrivial S] (h : IsAdjoinRootMonic S f) (x : R) :
    h.coeff (algebraMap R S x) = Pi.single 0 x := by
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    f : Polynomial R
    inst✝¹ : Algebra R S
    inst✝ : Nontrivial S
    h : IsAdjoinRootMonic S f
    x : R
    ⊢ Eq (h.coeff ((algebraMap R S) x)) (Pi.single 0 x)
  -/
  ext i
  /-
    case h
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    f : Polynomial R
    inst✝¹ : Algebra R S
    inst✝ : Nontrivial S
    h : IsAdjoinRootMonic S f
    x : R
    i : Nat
    ⊢ Eq (h.coeff ((algebraMap R S) x) i) (Pi.single 0 x i)
  -/
  rw [Algebra.algebraMap_eq_smul_one, map_smul, coeff_one, Pi.smul_apply, smul_eq_mul]
  /-
    case h
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    f : Polynomial R
    inst✝¹ : Algebra R S
    inst✝ : Nontrivial S
    h : IsAdjoinRootMonic S f
    x : R
    i : Nat
    ⊢ Eq (HMul.hMul x (Pi.single 0 1 i)) (Pi.single 0 x i)
  -/
  refine (Pi.apply_single (fun _ y => x * y) ?_ 0 1 i).trans (by simp)
  /-
    case h
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    f : Polynomial R
    inst✝¹ : Algebra R S
    inst✝ : Nontrivial S
    h : IsAdjoinRootMonic S f
    x : R
    i : Nat
    ⊢ ∀ (i : Nat), Eq ((fun x_1 y => HMul.hMul x y) i 0) 0
  -/
  simp
  /-
    🎉 no goals
  -/


theorem ext_elem (h : IsAdjoinRootMonic S f) ⦃x y : S⦄
    (hxy : ∀ i < natDegree f, h.coeff x i = h.coeff y i) : x = y :=
  EquivLike.injective h.basis.equivFun <|
    funext fun i => by
      rw [Basis.equivFun_apply, ← h.coeff_apply_coe, Basis.equivFun_apply, ← h.coeff_apply_coe,
        hxy i i.prop]


theorem ext_elem_iff (h : IsAdjoinRootMonic S f) {x y : S} :
    x = y ↔ ∀ i < natDegree f, h.coeff x i = h.coeff y i :=
  ⟨fun hxy _ _=> hxy ▸ rfl, fun hxy => h.ext_elem hxy⟩


theorem coeff_injective (h : IsAdjoinRootMonic S f) : Function.Injective h.coeff := fun _ _ hxy =>
  h.ext_elem fun _ _ => hxy ▸ rfl


theorem isIntegral_root (h : IsAdjoinRootMonic S f) : IsIntegral R h.root :=
  ⟨f, h.Monic, h.aeval_root⟩


@[simp]
theorem lift_self_apply (h : IsAdjoinRoot S f) (x : S) :
    h.lift (algebraMap R S) h.root h.aeval_root x = x := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    f : Polynomial R
    h : IsAdjoinRoot S f
    x : S
    ⊢ Eq ((IsAdjoinRoot.lift (algebraMap R S) h.root h ⋯) x) x
  -/
  rw [← h.map_repr x, lift_map, ← aeval_def, h.aeval_eq]
  /-
    🎉 no goals
  -/


theorem lift_self (h : IsAdjoinRoot S f) :
    h.lift (algebraMap R S) h.root h.aeval_root = RingHom.id S :=
  RingHom.ext h.lift_self_apply


/-- Adjoining a root gives a unique ring up to algebra isomorphism.

This is the converse of `IsAdjoinRoot.ofEquiv`: this turns an `IsAdjoinRoot` into an
`AlgEquiv`, and `IsAdjoinRoot.ofEquiv` turns an `AlgEquiv` into an `IsAdjoinRoot`.
-/
def aequiv (h : IsAdjoinRoot S f) (h' : IsAdjoinRoot T f) : S ≃ₐ[R] T :=
  { h.liftHom h'.root h'.aeval_root with
    toFun := h.liftHom h'.root h'.aeval_root
    invFun := h'.liftHom h.root h.aeval_root
                            /-
                              R : Type u
                              S : Type v
                              inst✝⁴ : CommRing R
                              inst✝³ : CommRing S
                              inst✝² : Algebra R S
                              f : Polynomial R
                              T : Type u_1
                              inst✝¹ : CommRing T
                              inst✝ : Algebra R T
                              h : IsAdjoinRoot S f
                              h' : IsAdjoinRoot T f
                              x : S
                              ⊢ Eq ((IsAdjoinRoot.liftHom h.root ⋯ h') ((IsAdjoinRoot.liftHom h'.root ⋯ h) x …
                            -/
    left_inv := fun x => by rw [← h.map_repr x, liftHom_map, aeval_eq, liftHom_map, aeval_eq]
                            /-
                              🎉 no goals
                            -/
                             /-
                               R : Type u
                               S : Type v
                               inst✝⁴ : CommRing R
                               inst✝³ : CommRing S
                               inst✝² : Algebra R S
                               f : Polynomial R
                               T : Type u_1
                               inst✝¹ : CommRing T
                               inst✝ : Algebra R T
                               h : IsAdjoinRoot S f
                               h' : IsAdjoinRoot T f
                               x : T
                               ⊢ Eq ((IsAdjoinRoot.liftHom h'.root ⋯ h) ((IsAdjoinRoot.liftHom h.root ⋯ h') x …
                             -/
    right_inv := fun x => by rw [← h'.map_repr x, liftHom_map, aeval_eq, liftHom_map, aeval_eq] }
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem aequiv_map (h : IsAdjoinRoot S f) (h' : IsAdjoinRoot T f) (z : R[X]) :
    h.aequiv h' (h.map z) = h'.map z := by
  /-
    R : Type u
    S : Type v
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    f : Polynomial R
    T : Type u_1
    inst✝¹ : CommRing T
    inst✝ : Algebra R T
    h : IsAdjoinRoot S f
    h' : IsAdjoinRoot T f
    z : Polynomial R
    ⊢ Eq ((h.aequiv h') (h.map z)) (h'.map z)
  -/
  rw [aequiv, AlgEquiv.coe_mk, Equiv.coe_fn_mk, liftHom_map, aeval_eq]
  /-
    🎉 no goals
  -/


@[simp]
theorem aequiv_root (h : IsAdjoinRoot S f) (h' : IsAdjoinRoot T f) :
    h.aequiv h' h.root = h'.root := by
  /-
    R : Type u
    S : Type v
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    f : Polynomial R
    T : Type u_1
    inst✝¹ : CommRing T
    inst✝ : Algebra R T
    h : IsAdjoinRoot S f
    h' : IsAdjoinRoot T f
    ⊢ Eq ((h.aequiv h') h.root) h'.root
  -/
  rw [aequiv, AlgEquiv.coe_mk, Equiv.coe_fn_mk, liftHom_root]
  /-
    🎉 no goals
  -/


@[simp]
theorem aequiv_self (h : IsAdjoinRoot S f) : h.aequiv h = AlgEquiv.refl := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    f : Polynomial R
    h : IsAdjoinRoot S f
    ⊢ Eq (h.aequiv h) AlgEquiv.refl
  -/
  ext a; exact h.lift_self_apply a
         /-
           🎉 no goals
         -/


@[simp]
theorem aequiv_symm (h : IsAdjoinRoot S f) (h' : IsAdjoinRoot T f) :
                                           /-
                                             R : Type u
                                             S : Type v
                                             inst✝⁴ : CommRing R
                                             inst✝³ : CommRing S
                                             inst✝² : Algebra R S
                                             f : Polynomial R
                                             T : Type u_1
                                             inst✝¹ : CommRing T
                                             inst✝ : Algebra R T
                                             h : IsAdjoinRoot S f
                                             h' : IsAdjoinRoot T f
                                             ⊢ Eq (h.aequiv h').symm (h'.aequiv h)
                                           -/
    (h.aequiv h').symm = h'.aequiv h := by ext; rfl
                                                /-
                                                  🎉 no goals
                                                -/


@[simp]
theorem lift_aequiv {U : Type*} [CommRing U] (h : IsAdjoinRoot S f) (h' : IsAdjoinRoot T f)
    (i : R →+* U) (x hx z) : h'.lift i x hx (h.aequiv h' z) = h.lift i x hx z := by
  /-
    R : Type u
    S : Type v
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R S
    f : Polynomial R
    T : Type u_1
    inst✝² : CommRing T
    inst✝¹ : Algebra R T
    U : Type u_2
    inst✝ : CommRing U
    h : IsAdjoinRoot S f
    h' : IsAdjoinRoot T f
    i : RingHom R U
    x : U
    hx : Eq (Polynomial.eval₂ i x f) 0
    z : S
    ⊢ Eq ((IsAdjoinRoot.lift i x h' hx) ((h.aequiv h') z)) ((IsAdjoinRoot.lift i x …
  -/
  rw [← h.map_repr z, aequiv_map, lift_map, lift_map]
  /-
    🎉 no goals
  -/


@[simp]
theorem liftHom_aequiv {U : Type*} [CommRing U] [Algebra R U] (h : IsAdjoinRoot S f)
    (h' : IsAdjoinRoot T f) (x : U) (hx z) : h'.liftHom x hx (h.aequiv h' z) = h.liftHom x hx z :=
  h.lift_aequiv h' _ _ hx _


@[simp]
theorem aequiv_aequiv {U : Type*} [CommRing U] [Algebra R U] (h : IsAdjoinRoot S f)
    (h' : IsAdjoinRoot T f) (h'' : IsAdjoinRoot U f) (x) :
    (h'.aequiv h'') (h.aequiv h' x) = h.aequiv h'' x :=
  h.liftHom_aequiv _ _ h''.aeval_root _


@[simp]
theorem aequiv_trans {U : Type*} [CommRing U] [Algebra R U] (h : IsAdjoinRoot S f)
    (h' : IsAdjoinRoot T f) (h'' : IsAdjoinRoot U f) :
                                                             /-
                                                               R : Type u
                                                               S : Type v
                                                               inst✝⁶ : CommRing R
                                                               inst✝⁵ : CommRing S
                                                               inst✝⁴ : Algebra R S
                                                               f : Polynomial R
                                                               T : Type u_1
                                                               inst✝³ : CommRing T
                                                               inst✝² : Algebra R T
                                                               U : Type u_2
                                                               inst✝¹ : CommRing U
                                                               inst✝ : Algebra R U
                                                               h : IsAdjoinRoot S f
                                                               h' : IsAdjoinRoot T f
                                                               h'' : IsAdjoinRoot U f
                                                               ⊢ Eq ((h.aequiv h').trans (h'.aequiv h'')) (h.aequiv h'')
                                                             -/
    (h.aequiv h').trans (h'.aequiv h'') = h.aequiv h'' := by ext z; exact h.aequiv_aequiv h' h'' z
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


/-- Transfer `IsAdjoinRoot` across an algebra isomorphism.

This is the converse of `IsAdjoinRoot.aequiv`: this turns an `AlgEquiv` into an `IsAdjoinRoot`,
and `IsAdjoinRoot.aequiv` turns an `IsAdjoinRoot` into an `AlgEquiv`.
-/
@[simps! map_apply]
def ofEquiv (h : IsAdjoinRoot S f) (e : S ≃ₐ[R] T) : IsAdjoinRoot T f where
  map := ((e : S ≃+* T) : S →+* T).comp h.map
  map_surjective := e.surjective.comp h.map_surjective
  ker_map := by
    /-
      R : Type u
      S : Type v
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      f : Polynomial R
      T : Type u_1
      inst✝¹ : CommRing T
      inst✝ : Algebra R T
      h : IsAdjoinRoot S f
      e : AlgEquiv R S T
      ⊢ Eq (RingHom.ker ((↑↑e).comp h.map)) (Ideal.span (Singleton.singleton f))
    -/
    rw [← RingHom.comap_ker, RingHom.ker_coe_equiv, ← RingHom.ker_eq_comap_bot, h.ker_map]
    /-
      🎉 no goals
    -/
  algebraMap_eq := by
    /-
      R : Type u
      S : Type v
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      f : Polynomial R
      T : Type u_1
      inst✝¹ : CommRing T
      inst✝ : Algebra R T
      h : IsAdjoinRoot S f
      e : AlgEquiv R S T
      ⊢ Eq (algebraMap R T) (((↑↑e).comp h.map).comp Polynomial.C)
    -/
    ext
    simp only [AlgEquiv.commutes, RingHom.comp_apply, AlgEquiv.coe_ringEquiv,
      RingEquiv.coe_toRingHom, ← h.algebraMap_apply]


@[simp]
theorem ofEquiv_root (h : IsAdjoinRoot S f) (e : S ≃ₐ[R] T) : (h.ofEquiv e).root = e h.root := rfl


@[simp]
theorem aequiv_ofEquiv {U : Type*} [CommRing U] [Algebra R U] (h : IsAdjoinRoot S f)
    (h' : IsAdjoinRoot T f) (e : T ≃ₐ[R] U) : h.aequiv (h'.ofEquiv e) = (h.aequiv h').trans e := by
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    f : Polynomial R
    T : Type u_1
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    U : Type u_2
    inst✝¹ : CommRing U
    inst✝ : Algebra R U
    h : IsAdjoinRoot S f
    h' : IsAdjoinRoot T f
    e : AlgEquiv R T U
    ⊢ Eq (h.aequiv (h'.ofEquiv e)) ((h.aequiv h').trans e)
  -/
  ext a; rw [← h.map_repr a, aequiv_map, AlgEquiv.trans_apply, aequiv_map, ofEquiv_map_apply]
         /-
           🎉 no goals
         -/


@[simp]
theorem ofEquiv_aequiv {U : Type*} [CommRing U] [Algebra R U] (h : IsAdjoinRoot S f)
    (h' : IsAdjoinRoot U f) (e : S ≃ₐ[R] T) :
    (h.ofEquiv e).aequiv h' = e.symm.trans (h.aequiv h') := by
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    f : Polynomial R
    T : Type u_1
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    U : Type u_2
    inst✝¹ : CommRing U
    inst✝ : Algebra R U
    h : IsAdjoinRoot S f
    h' : IsAdjoinRoot U f
    e : AlgEquiv R S T
    ⊢ Eq ((h.ofEquiv e).aequiv h') (e.symm.trans (h.aequiv h'))
  -/
  ext a
  rw [← (h.ofEquiv e).map_repr a, aequiv_map, AlgEquiv.trans_apply, ofEquiv_map_apply,
    e.symm_apply_apply, aequiv_map]


theorem minpoly_eq [IsDomain R] [IsDomain S] [NoZeroSMulDivisors R S] [IsIntegrallyClosed R]
    (h : IsAdjoinRootMonic S f) (hirr : Irreducible f) : minpoly R h.root = f :=
  let ⟨q, hq⟩ := minpoly.isIntegrallyClosed_dvd h.isIntegral_root h.aeval_root
  symm <|
    eq_of_monic_of_associated h.Monic (minpoly.monic h.isIntegral_root) <| by
      convert
        Associated.mul_left (minpoly R h.root) <|
          associated_one_iff_isUnit.2 <|
            (hirr.isUnit_or_isUnit hq).resolve_left <| minpoly.not_isUnit R h.root
      /-
        case h.e'_4
        R : Type u
        S : Type v
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        f : Polynomial R
        inst✝³ : IsDomain R
        inst✝² : IsDomain S
        inst✝¹ : NoZeroSMulDivisors R S
        inst✝ : IsIntegrallyClosed R
        h : IsAdjoinRootMonic S f
        hirr : Irreducible f
        q : Polynomial R
        hq : Eq f (HMul.hMul (minpoly R h.root) q)
        ⊢ Eq (minpoly R h.root) (HMul.hMul (minpoly R h.root) 1)
      -/
      rw [mul_one]
      /-
        🎉 no goals
      -/


theorem Algebra.adjoin.powerBasis'_minpoly_gen [IsDomain R] [IsDomain S] [NoZeroSMulDivisors R S]
    [IsIntegrallyClosed R] {x : S} (hx' : IsIntegral R x) :
    minpoly R x = minpoly R (Algebra.adjoin.powerBasis' hx').gen := by
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    inst✝³ : IsDomain R
    inst✝² : IsDomain S
    inst✝¹ : NoZeroSMulDivisors R S
    inst✝ : IsIntegrallyClosed R
    x : S
    hx' : IsIntegral R x
    ⊢ Eq (minpoly R x) (minpoly R (Algebra.adjoin.powerBasis' hx').gen)
  -/
  haveI := isDomain_of_prime (prime_of_isIntegrallyClosed hx')
  haveI :=
    noZeroSMulDivisors_of_prime_of_degree_ne_zero (prime_of_isIntegrallyClosed hx')
      (ne_of_lt (degree_pos hx')).symm
  rw [← minpolyGen_eq, adjoin.powerBasis', minpolyGen_map, minpolyGen_eq,
    AdjoinRoot.powerBasis'_gen, ← isAdjoinRootMonic_root_eq_root _ (monic hx'), minpoly_eq]
  /-
    case hirr
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    inst✝³ : IsDomain R
    inst✝² : IsDomain S
    inst✝¹ : NoZeroSMulDivisors R S
    inst✝ : IsIntegrallyClosed R
    x : S
    hx' : IsIntegral R x
    this✝ : IsDomain (AdjoinRoot (minpoly R x))
    this : NoZeroSMulDivisors R (AdjoinRoot (minpoly R x))
    ⊢ Irreducible (minpoly R x)
  -/
  exact irreducible hx'
  /-
    🎉 no goals
  -/


