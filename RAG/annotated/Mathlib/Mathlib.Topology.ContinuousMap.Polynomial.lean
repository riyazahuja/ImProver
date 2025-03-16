/--
Every polynomial with coefficients in a topological semiring gives a (bundled) continuous function.
-/
@[simps]
def toContinuousMap (p : R[X]) : C(R, R) :=
                             /-
                               R : Type u_1
                               inst✝² : Semiring R
                               inst✝¹ : TopologicalSpace R
                               inst✝ : TopologicalSemiring R
                               p : Polynomial R
                               ⊢ Continuous fun x => Polynomial.eval x p
                             -/
  ⟨fun x : R => p.eval x, by fun_prop⟩
                             /-
                               🎉 no goals
                             -/


open ContinuousMap in
lemma toContinuousMap_X_eq_id : X.toContinuousMap = .id R := by
  /-
    R : Type u_1
    inst✝² : Semiring R
    inst✝¹ : TopologicalSpace R
    inst✝ : TopologicalSemiring R
    ⊢ Eq Polynomial.X.toContinuousMap (ContinuousMap.id R)
  -/
  ext; simp
       /-
         🎉 no goals
       -/


/-- A polynomial as a continuous function,
with domain restricted to some subset of the semiring of coefficients.

(This is particularly useful when restricting to compact sets, e.g. `[0,1]`.)
-/
@[simps]
def toContinuousMapOn (p : R[X]) (X : Set R) : C(X, R) :=
                                        /-
                                          R : Type u_1
                                          inst✝² : Semiring R
                                          inst✝¹ : TopologicalSpace R
                                          inst✝ : TopologicalSemiring R
                                          p : Polynomial R
                                          X : Set R
                                          ⊢ Continuous fun x => p.toContinuousMap ↑x
                                        -/
  ⟨fun x : X => p.toContinuousMap x, by fun_prop⟩
                                        /-
                                          🎉 no goals
                                        -/


open ContinuousMap in
lemma toContinuousMapOn_X_eq_restrict_id (s : Set R) :
    X.toContinuousMapOn s = restrict s (.id R) := by
  /-
    R : Type u_1
    inst✝² : Semiring R
    inst✝¹ : TopologicalSpace R
    inst✝ : TopologicalSemiring R
    s : Set R
    ⊢ Eq (Polynomial.X.toContinuousMapOn s) (ContinuousMap.restrict s (ContinuousM …
  -/
  ext; simp
       /-
         🎉 no goals
       -/


-- TODO some lemmas about when `toContinuousMapOn` is injective?

@[simp]
theorem aeval_continuousMap_apply (g : R[X]) (f : C(α, R)) (x : α) :
    ((Polynomial.aeval f) g) x = g.eval (f x) := by
  /-
    R : Type u_1
    α : Type u_2
    inst✝³ : TopologicalSpace α
    inst✝² : CommSemiring R
    inst✝¹ : TopologicalSpace R
    inst✝ : TopologicalSemiring R
    g : Polynomial R
    f : ContinuousMap α R
    x : α
    ⊢ Eq (((Polynomial.aeval f) g) x) (Polynomial.eval (f x) g)
  -/
  refine Polynomial.induction_on' g ?_ ?_
    /-
      case refine_1
      R : Type u_1
      α : Type u_2
      inst✝³ : TopologicalSpace α
      inst✝² : CommSemiring R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalSemiring R
      g : Polynomial R
      f : ContinuousMap α R
      x : α
      ⊢ ∀ (p q : Polynomial R), Eq (((Polynomial.aeval f) p) x) (Polynomial.eval (f  …
    -/
  · intro p q hp hq
    /-
      case refine_1
      R : Type u_1
      α : Type u_2
      inst✝³ : TopologicalSpace α
      inst✝² : CommSemiring R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalSemiring R
      g : Polynomial R
      f : ContinuousMap α R
      x : α
      p q : Polynomial R
      hp : Eq (((Polynomial.aeval f) p) x) (Polynomial.eval (f x) p)
      hq : Eq (((Polynomial.aeval f) q) x) (Polynomial.eval (f x) q)
      ⊢ Eq (((Polynomial.aeval f) (HAdd.hAdd p q)) x) (Polynomial.eval (f x) (HAdd.h …
    -/
    simp [hp, hq]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      α : Type u_2
      inst✝³ : TopologicalSpace α
      inst✝² : CommSemiring R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalSemiring R
      g : Polynomial R
      f : ContinuousMap α R
      x : α
      ⊢ ∀ (n : Nat) (a : R), Eq (((Polynomial.aeval f) ((Polynomial.monomial n) a))  …
    -/
  · intro n a
    /-
      case refine_2
      R : Type u_1
      α : Type u_2
      inst✝³ : TopologicalSpace α
      inst✝² : CommSemiring R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalSemiring R
      g : Polynomial R
      f : ContinuousMap α R
      x : α
      n : Nat
      a : R
      ⊢ Eq (((Polynomial.aeval f) ((Polynomial.monomial n) a)) x) (Polynomial.eval ( …
    -/
    simp [Pi.pow_apply]
    /-
      🎉 no goals
    -/


/-- The algebra map from `R[X]` to continuous functions `C(R, R)`.
-/
@[simps]
def toContinuousMapAlgHom : R[X] →ₐ[R] C(R, R) where
  toFun p := p.toContinuousMap
  map_zero' := by
    /-
      R : Type u_1
      inst✝² : CommSemiring R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalSemiring R
      ⊢ Eq ((↑{ toFun := fun p => p.toContinuousMap, map_one' := ⋯, map_mul' := ⋯ }) …
    -/
    ext
    /-
      case h
      R : Type u_1
      inst✝² : CommSemiring R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalSemiring R
      a✝ : R
      ⊢ Eq (((↑{ toFun := fun p => p.toContinuousMap, map_one' := ⋯, map_mul' := ⋯ } …
    -/
    simp
    /-
      🎉 no goals
    -/
  map_add' _ _ := by
    /-
      R : Type u_1
      inst✝² : CommSemiring R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalSemiring R
      ⊢ Eq ((fun p => p.toContinuousMap) 1) 1
    -/
    /-
      R : Type u_1
      inst✝² : CommSemiring R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalSemiring R
      x✝¹ x✝ : Polynomial R
      ⊢ Eq ((↑{ toFun := fun p => p.toContinuousMap, map_one' := ⋯, map_mul' := ⋯ }) …
    -/
    /-
      case h
      R : Type u_1
      inst✝² : CommSemiring R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalSemiring R
      a✝ : R
      ⊢ Eq (((fun p => p.toContinuousMap) 1) a✝) (1 a✝)
    -/
    ext
    /-
      🎉 no goals
    -/
    /-
      case h
      R : Type u_1
      inst✝² : CommSemiring R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalSemiring R
      x✝¹ x✝ : Polynomial R
      a✝ : R
      ⊢ Eq (((↑{ toFun := fun p => p.toContinuousMap, map_one' := ⋯, map_mul' := ⋯ } …
    -/
    /-
      R : Type u_1
      inst✝² : CommSemiring R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalSemiring R
      x✝¹ x✝ : Polynomial R
      ⊢ Eq ({ toFun := fun p => p.toContinuousMap, map_one' := ⋯ }.toFun (HMul.hMul  …
    -/
    simp
    /-
      case h
      R : Type u_1
      inst✝² : CommSemiring R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalSemiring R
      x✝¹ x✝ : Polynomial R
      a✝ : R
      ⊢ Eq (({ toFun := fun p => p.toContinuousMap, map_one' := ⋯ }.toFun (HMul.hMul …
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
  map_one' := by
    ext
    simp
  map_mul' _ _ := by
    ext
    simp
  commutes' _ := by
    /-
      R : Type u_1
      inst✝² : CommSemiring R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalSemiring R
      x✝ : R
      ⊢ Eq ((↑↑{ toFun := fun p => p.toContinuousMap, map_one' := ⋯, map_mul' := ⋯,  …
    -/
    ext
    /-
      case h
      R : Type u_1
      inst✝² : CommSemiring R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalSemiring R
      x✝ a✝ : R
      ⊢ Eq (((↑↑{ toFun := fun p => p.toContinuousMap, map_one' := ⋯, map_mul' := ⋯, …
    -/
    simp [Algebra.algebraMap_eq_smul_one]
    /-
      🎉 no goals
    -/


/-- The algebra map from `R[X]` to continuous functions `C(X, R)`, for any subset `X` of `R`.
-/
@[simps]
def toContinuousMapOnAlgHom (X : Set R) : R[X] →ₐ[R] C(X, R) where
  toFun p := p.toContinuousMapOn X
  map_zero' := by
    /-
      R : Type u_1
      inst✝² : CommSemiring R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalSemiring R
      X : Set R
      ⊢ Eq ((↑{ toFun := fun p => p.toContinuousMapOn X, map_one' := ⋯, map_mul' :=  …
    -/
    ext
    /-
      case h
      R : Type u_1
      inst✝² : CommSemiring R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalSemiring R
      X : Set R
      a✝ : ↑X
      ⊢ Eq (((↑{ toFun := fun p => p.toContinuousMapOn X, map_one' := ⋯, map_mul' := …
    -/
    simp
    /-
      🎉 no goals
    -/
  map_add' _ _ := by
    /-
      R : Type u_1
      inst✝² : CommSemiring R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalSemiring R
      X : Set R
      ⊢ Eq ((fun p => p.toContinuousMapOn X) 1) 1
    -/
    /-
      R : Type u_1
      inst✝² : CommSemiring R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalSemiring R
      X : Set R
      x✝¹ x✝ : Polynomial R
      ⊢ Eq ((↑{ toFun := fun p => p.toContinuousMapOn X, map_one' := ⋯, map_mul' :=  …
    -/
    /-
      case h
      R : Type u_1
      inst✝² : CommSemiring R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalSemiring R
      X : Set R
      a✝ : ↑X
      ⊢ Eq (((fun p => p.toContinuousMapOn X) 1) a✝) (1 a✝)
    -/
    ext
    /-
      🎉 no goals
    -/
    /-
      case h
      R : Type u_1
      inst✝² : CommSemiring R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalSemiring R
      X : Set R
      x✝¹ x✝ : Polynomial R
      a✝ : ↑X
      ⊢ Eq (((↑{ toFun := fun p => p.toContinuousMapOn X, map_one' := ⋯, map_mul' := …
    -/
    /-
      R : Type u_1
      inst✝² : CommSemiring R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalSemiring R
      X : Set R
      x✝¹ x✝ : Polynomial R
      ⊢ Eq ({ toFun := fun p => p.toContinuousMapOn X, map_one' := ⋯ }.toFun (HMul.h …
    -/
    simp
    /-
      case h
      R : Type u_1
      inst✝² : CommSemiring R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalSemiring R
      X : Set R
      x✝¹ x✝ : Polynomial R
      a✝ : ↑X
      ⊢ Eq (({ toFun := fun p => p.toContinuousMapOn X, map_one' := ⋯ }.toFun (HMul. …
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
  map_one' := by
    ext
    simp
  map_mul' _ _ := by
    ext
    simp
  commutes' _ := by
    /-
      R : Type u_1
      inst✝² : CommSemiring R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalSemiring R
      X : Set R
      x✝ : R
      ⊢ Eq ((↑↑{ toFun := fun p => p.toContinuousMapOn X, map_one' := ⋯, map_mul' := …
    -/
    ext
    /-
      case h
      R : Type u_1
      inst✝² : CommSemiring R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalSemiring R
      X : Set R
      x✝ : R
      a✝ : ↑X
      ⊢ Eq (((↑↑{ toFun := fun p => p.toContinuousMapOn X, map_one' := ⋯, map_mul' : …
    -/
    simp [Algebra.algebraMap_eq_smul_one]
    /-
      🎉 no goals
    -/


/--
The subalgebra of polynomial functions in `C(X, R)`, for `X` a subset of some topological semiring
`R`.
-/
noncomputable -- Porting note: added noncomputable
def polynomialFunctions (X : Set R) : Subalgebra R C(X, R) :=
  (⊤ : Subalgebra R R[X]).map (Polynomial.toContinuousMapOnAlgHom X)


@[simp]
theorem polynomialFunctions_coe (X : Set R) :
    (polynomialFunctions X : Set C(X, R)) = Set.range (Polynomial.toContinuousMapOnAlgHom X) := by
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    inst✝¹ : TopologicalSpace R
    inst✝ : TopologicalSemiring R
    X : Set R
    ⊢ Eq (↑(polynomialFunctions X)) (Set.range ⇑(Polynomial.toContinuousMapOnAlgHo …
  -/
  ext
  /-
    case h
    R : Type u_1
    inst✝² : CommSemiring R
    inst✝¹ : TopologicalSpace R
    inst✝ : TopologicalSemiring R
    X : Set R
    x✝ : ContinuousMap (↑X) R
    ⊢ Iff (Membership.mem (↑(polynomialFunctions X)) x✝) (Membership.mem (Set.rang …
  -/
  simp [polynomialFunctions]
  /-
    🎉 no goals
  -/

-- TODO:
-- if `f : R → R` is an affine equivalence, then pulling back along `f`
-- induces a normed algebra isomorphism between `polynomialFunctions X` and
-- `polynomialFunctions (f ⁻¹' X)`, intertwining the pullback along `f` of `C(R, R)` to itself.

theorem polynomialFunctions_separatesPoints (X : Set R) : (polynomialFunctions X).SeparatesPoints :=
  fun x y h => by
  -- We use `Polynomial.X`, then clean up.
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    inst✝¹ : TopologicalSpace R
    inst✝ : TopologicalSemiring R
    X : Set R
    x y : ↑X
    h : Ne x y
    ⊢ Exists fun f => And (Membership.mem (Set.image (fun f => ⇑f) ↑(polynomialFun …
  -/
  refine ⟨_, ⟨⟨_, ⟨⟨Polynomial.X, ⟨Algebra.mem_top, rfl⟩⟩, rfl⟩⟩, ?_⟩⟩
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    inst✝¹ : TopologicalSpace R
    inst✝ : TopologicalSemiring R
    X : Set R
    x y : ↑X
    h : Ne x y
    ⊢ Ne ((fun f => ⇑f) (↑(Polynomial.toContinuousMapOnAlgHom X) Polynomial.X) x)  …
  -/
  dsimp; simp only [Polynomial.eval_X]
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    inst✝¹ : TopologicalSpace R
    inst✝ : TopologicalSemiring R
    X : Set R
    x y : ↑X
    h : Ne x y
    ⊢ Not (Eq ↑x ↑y)
  -/
  exact fun h' => h (Subtype.ext h')
  /-
    🎉 no goals
  -/


/-- The preimage of polynomials on `[0,1]` under the pullback map by `x ↦ (b-a) * x + a`
is the polynomials on `[a,b]`. -/
theorem polynomialFunctions.comap_compRightAlgHom_iccHomeoI (a b : ℝ) (h : a < b) :
    (polynomialFunctions I).comap (compRightAlgHom ℝ ℝ (iccHomeoI a b h).symm) =
      polynomialFunctions (Set.Icc a b) := by
  /-
    a b : Real
    h : LT.lt a b
    ⊢ Eq (Subalgebra.comap (ContinuousMap.compRightAlgHom Real Real ↑(iccHomeoI a  …
  -/
  ext f
  /-
    case h
    a b : Real
    h : LT.lt a b
    f : ContinuousMap (↑(Set.Icc a b)) Real
    ⊢ Iff (Membership.mem (Subalgebra.comap (ContinuousMap.compRightAlgHom Real Re …
  -/
  fconstructor
    /-
      case h.mp
      a b : Real
      h : LT.lt a b
      f : ContinuousMap (↑(Set.Icc a b)) Real
      ⊢ Membership.mem (Subalgebra.comap (ContinuousMap.compRightAlgHom Real Real ↑( …
    -/
  · rintro ⟨p, ⟨-, w⟩⟩
    /-
      case h.mp.intro.intro
      a b : Real
      h : LT.lt a b
      f : ContinuousMap (↑(Set.Icc a b)) Real
      p : Polynomial Real
      w : Eq (↑(Polynomial.toContinuousMapOnAlgHom unitInterval) p) (↑(ContinuousMap …
      ⊢ Membership.mem (polynomialFunctions (Set.Icc a b)) f
    -/
    rw [DFunLike.ext_iff] at w
    /-
      case h.mp.intro.intro
      a b : Real
      h : LT.lt a b
      f : ContinuousMap (↑(Set.Icc a b)) Real
      p : Polynomial Real
      w : ∀ (x : ↑unitInterval), Eq ((↑(Polynomial.toContinuousMapOnAlgHom unitInter …
      ⊢ Membership.mem (polynomialFunctions (Set.Icc a b)) f
    -/
    dsimp at w
    /-
      case h.mp.intro.intro
      a b : Real
      h : LT.lt a b
      f : ContinuousMap (↑(Set.Icc a b)) Real
      p : Polynomial Real
      w : ∀ (x : ↑unitInterval), Eq (Polynomial.eval (↑x) p) (f ((iccHomeoI a b h).s …
      ⊢ Membership.mem (polynomialFunctions (Set.Icc a b)) f
    -/
    let q := p.comp ((b - a)⁻¹ • Polynomial.X + Polynomial.C (-a * (b - a)⁻¹))
    /-
      case h.mp.intro.intro
      a b : Real
      h : LT.lt a b
      f : ContinuousMap (↑(Set.Icc a b)) Real
      p : Polynomial Real
      w : ∀ (x : ↑unitInterval), Eq (Polynomial.eval (↑x) p) (f ((iccHomeoI a b h).s …
      q : Polynomial Real := p.comp (HAdd.hAdd (HSMul.hSMul (Inv.inv (HSub.hSub b a) …
      ⊢ Membership.mem (polynomialFunctions (Set.Icc a b)) f
    -/
    refine ⟨q, ⟨?_, ?_⟩⟩
      /-
        case h.mp.intro.intro.refine_1
        a b : Real
        h : LT.lt a b
        f : ContinuousMap (↑(Set.Icc a b)) Real
        p : Polynomial Real
        w : ∀ (x : ↑unitInterval), Eq (Polynomial.eval (↑x) p) (f ((iccHomeoI a b h).s …
        q : Polynomial Real := p.comp (HAdd.hAdd (HSMul.hSMul (Inv.inv (HSub.hSub b a) …
        ⊢ Membership.mem (↑Top.top.toSubsemiring) q
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case h.mp.intro.intro.refine_2
        a b : Real
        h : LT.lt a b
        f : ContinuousMap (↑(Set.Icc a b)) Real
        p : Polynomial Real
        w : ∀ (x : ↑unitInterval), Eq (Polynomial.eval (↑x) p) (f ((iccHomeoI a b h).s …
        q : Polynomial Real := p.comp (HAdd.hAdd (HSMul.hSMul (Inv.inv (HSub.hSub b a) …
        ⊢ Eq (↑(Polynomial.toContinuousMapOnAlgHom (Set.Icc a b)) q) f
      -/
    · ext x
      simp only [q, neg_mul, RingHom.map_neg, RingHom.map_mul, AlgHom.coe_toRingHom,
        Polynomial.eval_X, Polynomial.eval_neg, Polynomial.eval_C, Polynomial.eval_smul,
        smul_eq_mul, Polynomial.eval_mul, Polynomial.eval_add, Polynomial.coe_aeval_eq_eval,
        Polynomial.eval_comp, Polynomial.toContinuousMapOnAlgHom_apply,
        Polynomial.toContinuousMapOn_apply, Polynomial.toContinuousMap_apply]
      /-
        case h.mp.intro.intro.refine_2.h
        a b : Real
        h : LT.lt a b
        f : ContinuousMap (↑(Set.Icc a b)) Real
        p : Polynomial Real
        w : ∀ (x : ↑unitInterval), Eq (Polynomial.eval (↑x) p) (f ((iccHomeoI a b h).s …
        q : Polynomial Real := p.comp (HAdd.hAdd (HSMul.hSMul (Inv.inv (HSub.hSub b a) …
        x : ↑(Set.Icc a b)
        ⊢ Eq (Polynomial.eval (HAdd.hAdd (HMul.hMul (Inv.inv (HSub.hSub b a)) ↑x) (Neg …
      -/
      convert w ⟨_, _⟩
        /-
          case h.e'_3.h.e'_6
          a b : Real
          h : LT.lt a b
          f : ContinuousMap (↑(Set.Icc a b)) Real
          p : Polynomial Real
          w : ∀ (x : ↑unitInterval), Eq (Polynomial.eval (↑x) p) (f ((iccHomeoI a b h).s …
          q : Polynomial Real := p.comp (HAdd.hAdd (HSMul.hSMul (Inv.inv (HSub.hSub b a) …
          x : ↑(Set.Icc a b)
          ⊢ Eq x ((iccHomeoI a b h).symm ⟨HAdd.hAdd (HMul.hMul (Inv.inv (HSub.hSub b a)) …
        -/
      · ext
        /-
          case h.e'_3.h.e'_6.a
          a b : Real
          h : LT.lt a b
          f : ContinuousMap (↑(Set.Icc a b)) Real
          p : Polynomial Real
          w : ∀ (x : ↑unitInterval), Eq (Polynomial.eval (↑x) p) (f ((iccHomeoI a b h).s …
          q : Polynomial Real := p.comp (HAdd.hAdd (HSMul.hSMul (Inv.inv (HSub.hSub b a) …
          x : ↑(Set.Icc a b)
          ⊢ Eq ↑x ↑((iccHomeoI a b h).symm ⟨HAdd.hAdd (HMul.hMul (Inv.inv (HSub.hSub b a …
        -/
        simp only [iccHomeoI_symm_apply_coe, Subtype.coe_mk]
        /-
          case h.e'_3.h.e'_6.a
          a b : Real
          h : LT.lt a b
          f : ContinuousMap (↑(Set.Icc a b)) Real
          p : Polynomial Real
          w : ∀ (x : ↑unitInterval), Eq (Polynomial.eval (↑x) p) (f ((iccHomeoI a b h).s …
          q : Polynomial Real := p.comp (HAdd.hAdd (HSMul.hSMul (Inv.inv (HSub.hSub b a) …
          x : ↑(Set.Icc a b)
          ⊢ Eq (↑x) (HAdd.hAdd (HMul.hMul (HSub.hSub b a) (HAdd.hAdd (HMul.hMul (Inv.inv …
        -/
        replace h : b - a ≠ 0 := sub_ne_zero_of_ne h.ne.symm
        /-
          case h.e'_3.h.e'_6.a
          a b : Real
          h✝ : LT.lt a b
          f : ContinuousMap (↑(Set.Icc a b)) Real
          p : Polynomial Real
          w : ∀ (x : ↑unitInterval), Eq (Polynomial.eval (↑x) p) (f ((iccHomeoI a b h✝). …
          q : Polynomial Real := p.comp (HAdd.hAdd (HSMul.hSMul (Inv.inv (HSub.hSub b a) …
          x : ↑(Set.Icc a b)
          h : Ne (HSub.hSub b a) 0
          ⊢ Eq (↑x) (HAdd.hAdd (HMul.hMul (HSub.hSub b a) (HAdd.hAdd (HMul.hMul (Inv.inv …
        -/
        simp only [mul_add]
        /-
          case h.e'_3.h.e'_6.a
          a b : Real
          h✝ : LT.lt a b
          f : ContinuousMap (↑(Set.Icc a b)) Real
          p : Polynomial Real
          w : ∀ (x : ↑unitInterval), Eq (Polynomial.eval (↑x) p) (f ((iccHomeoI a b h✝). …
          q : Polynomial Real := p.comp (HAdd.hAdd (HSMul.hSMul (Inv.inv (HSub.hSub b a) …
          x : ↑(Set.Icc a b)
          h : Ne (HSub.hSub b a) 0
          ⊢ Eq (↑x) (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HSub.hSub b a) (HMul.hMul (Inv.inv …
        -/
        field_simp
        /-
          case h.e'_3.h.e'_6.a
          a b : Real
          h✝ : LT.lt a b
          f : ContinuousMap (↑(Set.Icc a b)) Real
          p : Polynomial Real
          w : ∀ (x : ↑unitInterval), Eq (Polynomial.eval (↑x) p) (f ((iccHomeoI a b h✝). …
          q : Polynomial Real := p.comp (HAdd.hAdd (HSMul.hSMul (Inv.inv (HSub.hSub b a) …
          x : ↑(Set.Icc a b)
          h : Ne (HSub.hSub b a) 0
          ⊢ Eq (HMul.hMul (↑x) (HSub.hSub b a)) (HAdd.hAdd (HAdd.hAdd (HMul.hMul (↑x) (H …
        -/
        ring
        /-
          🎉 no goals
        -/
        /-
          case h.mp.intro.intro.refine_2.h.convert_2
          a b : Real
          h : LT.lt a b
          f : ContinuousMap (↑(Set.Icc a b)) Real
          p : Polynomial Real
          w : ∀ (x : ↑unitInterval), Eq (Polynomial.eval (↑x) p) (f ((iccHomeoI a b h).s …
          q : Polynomial Real := p.comp (HAdd.hAdd (HSMul.hSMul (Inv.inv (HSub.hSub b a) …
          x : ↑(Set.Icc a b)
          ⊢ Membership.mem unitInterval (HAdd.hAdd (HMul.hMul (Inv.inv (HSub.hSub b a))  …
        -/
      · change _ + _ ∈ I
        /-
          case h.mp.intro.intro.refine_2.h.convert_2
          a b : Real
          h : LT.lt a b
          f : ContinuousMap (↑(Set.Icc a b)) Real
          p : Polynomial Real
          w : ∀ (x : ↑unitInterval), Eq (Polynomial.eval (↑x) p) (f ((iccHomeoI a b h).s …
          q : Polynomial Real := p.comp (HAdd.hAdd (HSMul.hSMul (Inv.inv (HSub.hSub b a) …
          x : ↑(Set.Icc a b)
          ⊢ Membership.mem unitInterval (HAdd.hAdd (HMul.hMul (Inv.inv (HSub.hSub b a))  …
        -/
        rw [mul_comm (b - a)⁻¹, ← neg_mul, ← add_mul, ← sub_eq_add_neg]
        /-
          case h.mp.intro.intro.refine_2.h.convert_2
          a b : Real
          h : LT.lt a b
          f : ContinuousMap (↑(Set.Icc a b)) Real
          p : Polynomial Real
          w : ∀ (x : ↑unitInterval), Eq (Polynomial.eval (↑x) p) (f ((iccHomeoI a b h).s …
          q : Polynomial Real := p.comp (HAdd.hAdd (HSMul.hSMul (Inv.inv (HSub.hSub b a) …
          x : ↑(Set.Icc a b)
          ⊢ Membership.mem unitInterval (HMul.hMul (HSub.hSub (↑x) a) (Inv.inv (HSub.hSu …
        -/
        have w₁ : 0 < (b - a)⁻¹ := inv_pos.mpr (sub_pos.mpr h)
        /-
          case h.mp.intro.intro.refine_2.h.convert_2
          a b : Real
          h : LT.lt a b
          f : ContinuousMap (↑(Set.Icc a b)) Real
          p : Polynomial Real
          w : ∀ (x : ↑unitInterval), Eq (Polynomial.eval (↑x) p) (f ((iccHomeoI a b h).s …
          q : Polynomial Real := p.comp (HAdd.hAdd (HSMul.hSMul (Inv.inv (HSub.hSub b a) …
          x : ↑(Set.Icc a b)
          w₁ : LT.lt 0 (Inv.inv (HSub.hSub b a))
          ⊢ Membership.mem unitInterval (HMul.hMul (HSub.hSub (↑x) a) (Inv.inv (HSub.hSu …
        -/
        have w₂ : 0 ≤ (x : ℝ) - a := sub_nonneg.mpr x.2.1
        /-
          case h.mp.intro.intro.refine_2.h.convert_2
          a b : Real
          h : LT.lt a b
          f : ContinuousMap (↑(Set.Icc a b)) Real
          p : Polynomial Real
          w : ∀ (x : ↑unitInterval), Eq (Polynomial.eval (↑x) p) (f ((iccHomeoI a b h).s …
          q : Polynomial Real := p.comp (HAdd.hAdd (HSMul.hSMul (Inv.inv (HSub.hSub b a) …
          x : ↑(Set.Icc a b)
          w₁ : LT.lt 0 (Inv.inv (HSub.hSub b a))
          w₂ : LE.le 0 (HSub.hSub (↑x) a)
          ⊢ Membership.mem unitInterval (HMul.hMul (HSub.hSub (↑x) a) (Inv.inv (HSub.hSu …
        -/
        have w₃ : (x : ℝ) - a ≤ b - a := sub_le_sub_right x.2.2 a
        /-
          case h.mp.intro.intro.refine_2.h.convert_2
          a b : Real
          h : LT.lt a b
          f : ContinuousMap (↑(Set.Icc a b)) Real
          p : Polynomial Real
          w : ∀ (x : ↑unitInterval), Eq (Polynomial.eval (↑x) p) (f ((iccHomeoI a b h).s …
          q : Polynomial Real := p.comp (HAdd.hAdd (HSMul.hSMul (Inv.inv (HSub.hSub b a) …
          x : ↑(Set.Icc a b)
          w₁ : LT.lt 0 (Inv.inv (HSub.hSub b a))
          w₂ : LE.le 0 (HSub.hSub (↑x) a)
          w₃ : LE.le (HSub.hSub (↑x) a) (HSub.hSub b a)
          ⊢ Membership.mem unitInterval (HMul.hMul (HSub.hSub (↑x) a) (Inv.inv (HSub.hSu …
        -/
        fconstructor
          /-
            case h.mp.intro.intro.refine_2.h.convert_2.left
            a b : Real
            h : LT.lt a b
            f : ContinuousMap (↑(Set.Icc a b)) Real
            p : Polynomial Real
            w : ∀ (x : ↑unitInterval), Eq (Polynomial.eval (↑x) p) (f ((iccHomeoI a b h).s …
            q : Polynomial Real := p.comp (HAdd.hAdd (HSMul.hSMul (Inv.inv (HSub.hSub b a) …
            x : ↑(Set.Icc a b)
            w₁ : LT.lt 0 (Inv.inv (HSub.hSub b a))
            w₂ : LE.le 0 (HSub.hSub (↑x) a)
            w₃ : LE.le (HSub.hSub (↑x) a) (HSub.hSub b a)
            ⊢ LE.le 0 (HMul.hMul (HSub.hSub (↑x) a) (Inv.inv (HSub.hSub b a)))
          -/
        · exact mul_nonneg w₂ (le_of_lt w₁)
          /-
            🎉 no goals
          -/
          /-
            case h.mp.intro.intro.refine_2.h.convert_2.right
            a b : Real
            h : LT.lt a b
            f : ContinuousMap (↑(Set.Icc a b)) Real
            p : Polynomial Real
            w : ∀ (x : ↑unitInterval), Eq (Polynomial.eval (↑x) p) (f ((iccHomeoI a b h).s …
            q : Polynomial Real := p.comp (HAdd.hAdd (HSMul.hSMul (Inv.inv (HSub.hSub b a) …
            x : ↑(Set.Icc a b)
            w₁ : LT.lt 0 (Inv.inv (HSub.hSub b a))
            w₂ : LE.le 0 (HSub.hSub (↑x) a)
            w₃ : LE.le (HSub.hSub (↑x) a) (HSub.hSub b a)
            ⊢ LE.le (HMul.hMul (HSub.hSub (↑x) a) (Inv.inv (HSub.hSub b a))) 1
          -/
        · rw [← div_eq_mul_inv, div_le_one (sub_pos.mpr h)]
          /-
            case h.mp.intro.intro.refine_2.h.convert_2.right
            a b : Real
            h : LT.lt a b
            f : ContinuousMap (↑(Set.Icc a b)) Real
            p : Polynomial Real
            w : ∀ (x : ↑unitInterval), Eq (Polynomial.eval (↑x) p) (f ((iccHomeoI a b h).s …
            q : Polynomial Real := p.comp (HAdd.hAdd (HSMul.hSMul (Inv.inv (HSub.hSub b a) …
            x : ↑(Set.Icc a b)
            w₁ : LT.lt 0 (Inv.inv (HSub.hSub b a))
            w₂ : LE.le 0 (HSub.hSub (↑x) a)
            w₃ : LE.le (HSub.hSub (↑x) a) (HSub.hSub b a)
            ⊢ LE.le (HSub.hSub (↑x) a) (HSub.hSub b a)
          -/
          exact w₃
          /-
            🎉 no goals
          -/
    /-
      case h.mpr
      a b : Real
      h : LT.lt a b
      f : ContinuousMap (↑(Set.Icc a b)) Real
      ⊢ Membership.mem (polynomialFunctions (Set.Icc a b)) f → Membership.mem (Subal …
    -/
  · rintro ⟨p, ⟨-, rfl⟩⟩
    /-
      case h.mpr.intro.intro
      a b : Real
      h : LT.lt a b
      p : Polynomial Real
      ⊢ Membership.mem (Subalgebra.comap (ContinuousMap.compRightAlgHom Real Real ↑( …
    -/
    let q := p.comp ((b - a) • Polynomial.X + Polynomial.C a)
    /-
      case h.mpr.intro.intro
      a b : Real
      h : LT.lt a b
      p : Polynomial Real
      q : Polynomial Real := p.comp (HAdd.hAdd (HSMul.hSMul (HSub.hSub b a) Polynomi …
      ⊢ Membership.mem (Subalgebra.comap (ContinuousMap.compRightAlgHom Real Real ↑( …
    -/
    refine ⟨q, ⟨?_, ?_⟩⟩
      /-
        case h.mpr.intro.intro.refine_1
        a b : Real
        h : LT.lt a b
        p : Polynomial Real
        q : Polynomial Real := p.comp (HAdd.hAdd (HSMul.hSMul (HSub.hSub b a) Polynomi …
        ⊢ Membership.mem (↑Top.top.toSubsemiring) q
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case h.mpr.intro.intro.refine_2
        a b : Real
        h : LT.lt a b
        p : Polynomial Real
        q : Polynomial Real := p.comp (HAdd.hAdd (HSMul.hSMul (HSub.hSub b a) Polynomi …
        ⊢ Eq (↑(Polynomial.toContinuousMapOnAlgHom unitInterval) q) (↑(ContinuousMap.c …
      -/
    · ext x
      /-
        case h.mpr.intro.intro.refine_2.h
        a b : Real
        h : LT.lt a b
        p : Polynomial Real
        q : Polynomial Real := p.comp (HAdd.hAdd (HSMul.hSMul (HSub.hSub b a) Polynomi …
        x : ↑unitInterval
        ⊢ Eq ((↑(Polynomial.toContinuousMapOnAlgHom unitInterval) q) x) ((↑(Continuous …
      -/
      simp [q, mul_comm]
      /-
        🎉 no goals
      -/


theorem polynomialFunctions.eq_adjoin_X (s : Set R) :
    polynomialFunctions s = Algebra.adjoin R {toContinuousMapOnAlgHom s X} := by
  refine le_antisymm ?_
    (Algebra.adjoin_le fun _ h => ⟨X, trivial, (Set.mem_singleton_iff.1 h).symm⟩)
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    inst✝¹ : TopologicalSpace R
    inst✝ : TopologicalSemiring R
    s : Set R
    ⊢ LE.le (polynomialFunctions s) (Algebra.adjoin R (Singleton.singleton ((Polyn …
  -/
  rintro - ⟨p, -, rfl⟩
  /-
    case intro.intro
    R : Type u_1
    inst✝² : CommSemiring R
    inst✝¹ : TopologicalSpace R
    inst✝ : TopologicalSemiring R
    s : Set R
    p : Polynomial R
    ⊢ Membership.mem (Algebra.adjoin R (Singleton.singleton ((Polynomial.toContinu …
  -/
  rw [AlgHom.coe_toRingHom]
  /-
    case intro.intro
    R : Type u_1
    inst✝² : CommSemiring R
    inst✝¹ : TopologicalSpace R
    inst✝ : TopologicalSemiring R
    s : Set R
    p : Polynomial R
    ⊢ Membership.mem (Algebra.adjoin R (Singleton.singleton ((Polynomial.toContinu …
  -/
  refine p.induction_on (fun r => ?_) (fun f g hf hg => ?_) fun n r hn => ?_
    /-
      case intro.intro.refine_1
      R : Type u_1
      inst✝² : CommSemiring R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalSemiring R
      s : Set R
      p : Polynomial R
      r : R
      ⊢ Membership.mem (Algebra.adjoin R (Singleton.singleton ((Polynomial.toContinu …
    -/
  · rw [Polynomial.C_eq_algebraMap, AlgHomClass.commutes]
    /-
      case intro.intro.refine_1
      R : Type u_1
      inst✝² : CommSemiring R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalSemiring R
      s : Set R
      p : Polynomial R
      r : R
      ⊢ Membership.mem (Algebra.adjoin R (Singleton.singleton ((Polynomial.toContinu …
    -/
    exact Subalgebra.algebraMap_mem _ r
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      R : Type u_1
      inst✝² : CommSemiring R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalSemiring R
      s : Set R
      p f g : Polynomial R
      hf : Membership.mem (Algebra.adjoin R (Singleton.singleton ((Polynomial.toCont …
      hg : Membership.mem (Algebra.adjoin R (Singleton.singleton ((Polynomial.toCont …
      ⊢ Membership.mem (Algebra.adjoin R (Singleton.singleton ((Polynomial.toContinu …
    -/
  · rw [map_add]
    /-
      case intro.intro.refine_2
      R : Type u_1
      inst✝² : CommSemiring R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalSemiring R
      s : Set R
      p f g : Polynomial R
      hf : Membership.mem (Algebra.adjoin R (Singleton.singleton ((Polynomial.toCont …
      hg : Membership.mem (Algebra.adjoin R (Singleton.singleton ((Polynomial.toCont …
      ⊢ Membership.mem (Algebra.adjoin R (Singleton.singleton ((Polynomial.toContinu …
    -/
    exact add_mem hf hg
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_3
      R : Type u_1
      inst✝² : CommSemiring R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalSemiring R
      s : Set R
      p : Polynomial R
      n : Nat
      r : R
      hn : Membership.mem (Algebra.adjoin R (Singleton.singleton ((Polynomial.toCont …
      ⊢ Membership.mem (Algebra.adjoin R (Singleton.singleton ((Polynomial.toContinu …
    -/
  · rw [pow_succ, ← mul_assoc, map_mul]
    /-
      case intro.intro.refine_3
      R : Type u_1
      inst✝² : CommSemiring R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalSemiring R
      s : Set R
      p : Polynomial R
      n : Nat
      r : R
      hn : Membership.mem (Algebra.adjoin R (Singleton.singleton ((Polynomial.toCont …
      ⊢ Membership.mem (Algebra.adjoin R (Singleton.singleton ((Polynomial.toContinu …
    -/
    exact mul_mem hn (Algebra.subset_adjoin <| Set.mem_singleton _)
    /-
      🎉 no goals
    -/


theorem polynomialFunctions.le_equalizer {A : Type*} [Semiring A] [Algebra R A] (s : Set R)
    (φ ψ : C(s, R) →ₐ[R] A)
    (h : φ (toContinuousMapOnAlgHom s X) = ψ (toContinuousMapOnAlgHom s X)) :
    polynomialFunctions s ≤ AlgHom.equalizer φ ψ := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    inst✝³ : TopologicalSpace R
    inst✝² : TopologicalSemiring R
    A : Type u_2
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    s : Set R
    φ ψ : AlgHom R (ContinuousMap (↑s) R) A
    h : Eq (φ ((Polynomial.toContinuousMapOnAlgHom s) Polynomial.X)) (ψ ((Polynomi …
    ⊢ LE.le (polynomialFunctions s) (AlgHom.equalizer φ ψ)
  -/
  rw [polynomialFunctions.eq_adjoin_X s]
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    inst✝³ : TopologicalSpace R
    inst✝² : TopologicalSemiring R
    A : Type u_2
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    s : Set R
    φ ψ : AlgHom R (ContinuousMap (↑s) R) A
    h : Eq (φ ((Polynomial.toContinuousMapOnAlgHom s) Polynomial.X)) (ψ ((Polynomi …
    ⊢ LE.le (Algebra.adjoin R (Singleton.singleton ((Polynomial.toContinuousMapOnA …
  -/
  exact φ.adjoin_le_equalizer ψ fun x hx => (Set.mem_singleton_iff.1 hx).symm ▸ h
  /-
    🎉 no goals
  -/


theorem polynomialFunctions.starClosure_eq_adjoin_X [StarRing R] [ContinuousStar R] (s : Set R) :
    (polynomialFunctions s).starClosure = adjoin R {toContinuousMapOnAlgHom s X} := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    inst✝³ : TopologicalSpace R
    inst✝² : TopologicalSemiring R
    inst✝¹ : StarRing R
    inst✝ : ContinuousStar R
    s : Set R
    ⊢ Eq (polynomialFunctions s).starClosure (StarAlgebra.adjoin R (Singleton.sing …
  -/
  rw [polynomialFunctions.eq_adjoin_X s, adjoin_eq_starClosure_adjoin]
  /-
    🎉 no goals
  -/


theorem polynomialFunctions.starClosure_le_equalizer {A : Type*} [StarRing R] [ContinuousStar R]
    [Semiring A] [StarRing A] [Algebra R A] (s : Set R) (φ ψ : C(s, R) →⋆ₐ[R] A)
    (h : φ (toContinuousMapOnAlgHom s X) = ψ (toContinuousMapOnAlgHom s X)) :
    (polynomialFunctions s).starClosure ≤ StarAlgHom.equalizer φ ψ := by
  /-
    R : Type u_1
    inst✝⁷ : CommSemiring R
    inst✝⁶ : TopologicalSpace R
    inst✝⁵ : TopologicalSemiring R
    A : Type u_2
    inst✝⁴ : StarRing R
    inst✝³ : ContinuousStar R
    inst✝² : Semiring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    s : Set R
    φ ψ : StarAlgHom R (ContinuousMap (↑s) R) A
    h : Eq (φ ((Polynomial.toContinuousMapOnAlgHom s) Polynomial.X)) (ψ ((Polynomi …
    ⊢ LE.le (polynomialFunctions s).starClosure (StarAlgHom.equalizer φ ψ)
  -/
  rw [polynomialFunctions.starClosure_eq_adjoin_X s]
  /-
    R : Type u_1
    inst✝⁷ : CommSemiring R
    inst✝⁶ : TopologicalSpace R
    inst✝⁵ : TopologicalSemiring R
    A : Type u_2
    inst✝⁴ : StarRing R
    inst✝³ : ContinuousStar R
    inst✝² : Semiring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    s : Set R
    φ ψ : StarAlgHom R (ContinuousMap (↑s) R) A
    h : Eq (φ ((Polynomial.toContinuousMapOnAlgHom s) Polynomial.X)) (ψ ((Polynomi …
    ⊢ LE.le (StarAlgebra.adjoin R (Singleton.singleton ((Polynomial.toContinuousMa …
  -/
  exact StarAlgHom.adjoin_le_equalizer φ ψ fun x hx => (Set.mem_singleton_iff.1 hx).symm ▸ h
  /-
    🎉 no goals
  -/


