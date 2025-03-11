theorem taylor_mem_nonZeroDivisors (hp : p ∈ R[X]⁰) : taylor r p ∈ R[X]⁰ := by
  /-
    R : Type u
    inst✝ : CommRing R
    r : R
    p : Polynomial R
    hp : Membership.mem (nonZeroDivisors (Polynomial R)) p
    ⊢ Membership.mem (nonZeroDivisors (Polynomial R)) ((Polynomial.taylor r) p)
  -/
  rw [mem_nonZeroDivisors_iff]
  /-
    R : Type u
    inst✝ : CommRing R
    r : R
    p : Polynomial R
    hp : Membership.mem (nonZeroDivisors (Polynomial R)) p
    ⊢ ∀ (x : Polynomial R), Eq (HMul.hMul x ((Polynomial.taylor r) p)) 0 → Eq x 0
  -/
  intro x hx
  /-
    R : Type u
    inst✝ : CommRing R
    r : R
    p : Polynomial R
    hp : Membership.mem (nonZeroDivisors (Polynomial R)) p
    x : Polynomial R
    hx : Eq (HMul.hMul x ((Polynomial.taylor r) p)) 0
    ⊢ Eq x 0
  -/
  have : x = taylor (r - r) x := by simp
  rwa [this, sub_eq_add_neg, ← taylor_taylor, ← taylor_mul,
    LinearMap.map_eq_zero_iff _ (taylor_injective _), mul_right_mem_nonZeroDivisors_eq_zero_iff hp,
    LinearMap.map_eq_zero_iff _ (taylor_injective _)] at hx


/-- The Laurent expansion of rational functions about a value.
Auxiliary definition, usage when over integral domains should prefer `RatFunc.laurent`. -/
def laurentAux : RatFunc R →+* RatFunc R :=
  RatFunc.mapRingHom
    ( { toFun := taylor r
        map_add' := map_add (taylor r)
        map_mul' := taylor_mul _
        map_zero' := map_zero (taylor r)
        map_one' := taylor_one r } : R[X] →+* R[X])
    (taylor_mem_nonZeroDivisors _)


theorem laurentAux_ofFractionRing_mk (q : R[X]⁰) :
    laurentAux r (ofFractionRing (Localization.mk p q)) =
      ofFractionRing (.mk (taylor r p) ⟨taylor r q, taylor_mem_nonZeroDivisors r q q.prop⟩) :=
  map_apply_ofFractionRing_mk _ _ _ _


theorem laurentAux_div :
    laurentAux r (algebraMap _ _ p / algebraMap _ _ q) =
      algebraMap _ _ (taylor r p) / algebraMap _ _ (taylor r q) :=
  -- Porting note: added `by exact taylor_mem_nonZeroDivisors r`
                      /-
                        R : Type u
                        inst✝¹ : CommRing R
                        r : R
                        p q : Polynomial R
                        inst✝ : IsDomain R
                        ⊢ LE.le (nonZeroDivisors (Polynomial R)) (Submonoid.comap { toFun := ⇑(Polynom …
                      -/
  map_apply_div _ (by exact taylor_mem_nonZeroDivisors r) _ _
                      /-
                        🎉 no goals
                      -/


@[simp]
theorem laurentAux_algebraMap : laurentAux r (algebraMap _ _ p) = algebraMap _ _ (taylor r p) := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    r : R
    p : Polynomial R
    inst✝ : IsDomain R
    ⊢ Eq ((RatFunc.laurentAux r) ((algebraMap (Polynomial R) (RatFunc R)) p)) ((al …
  -/
  rw [← mk_one, ← mk_one, mk_eq_div, laurentAux_div, mk_eq_div, taylor_one, map_one, map_one]
  /-
    🎉 no goals
  -/


/-- The Laurent expansion of rational functions about a value. -/
def laurent : RatFunc R →ₐ[R] RatFunc R :=
  RatFunc.mapAlgHom (.ofLinearMap (taylor r) (taylor_one _) (taylor_mul _))
    (taylor_mem_nonZeroDivisors _)


theorem laurent_div :
    laurent r (algebraMap _ _ p / algebraMap _ _ q) =
      algebraMap _ _ (taylor r p) / algebraMap _ _ (taylor r q) :=
  laurentAux_div r p q


@[simp]
theorem laurent_algebraMap : laurent r (algebraMap _ _ p) = algebraMap _ _ (taylor r p) :=
  laurentAux_algebraMap _ _


@[simp]
theorem laurent_X : laurent r X = X + C r := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    r : R
    inst✝ : IsDomain R
    ⊢ Eq ((RatFunc.laurent r) RatFunc.X) (HAdd.hAdd RatFunc.X (RatFunc.C r))
  -/
  rw [← algebraMap_X, laurent_algebraMap, taylor_X, _root_.map_add, algebraMap_C]
  /-
    🎉 no goals
  -/


@[simp]
theorem laurent_C (x : R) : laurent r (C x) = C x := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    r : R
    inst✝ : IsDomain R
    x : R
    ⊢ Eq ((RatFunc.laurent r) (RatFunc.C x)) (RatFunc.C x)
  -/
  rw [← algebraMap_C, laurent_algebraMap, taylor_C]
  /-
    🎉 no goals
  -/


@[simp]
                                                /-
                                                  R : Type u
                                                  inst✝¹ : CommRing R
                                                  f : RatFunc R
                                                  inst✝ : IsDomain R
                                                  ⊢ Eq ((RatFunc.laurent 0) f) f
                                                -/
theorem laurent_at_zero : laurent 0 f = f := by induction f using RatFunc.induction_on; simp
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/


theorem laurent_laurent : laurent r (laurent s f) = laurent (r + s) f := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    r s : R
    f : RatFunc R
    inst✝ : IsDomain R
    ⊢ Eq ((RatFunc.laurent r) ((RatFunc.laurent s) f)) ((RatFunc.laurent (HAdd.hAd …
  -/
  induction f using RatFunc.induction_on
  /-
    case f
    R : Type u
    inst✝¹ : CommRing R
    r s : R
    f : RatFunc R
    inst✝ : IsDomain R
    p✝ q✝ : Polynomial R
    x✝ : Ne q✝ 0
    ⊢ Eq ((RatFunc.laurent r) ((RatFunc.laurent s) (HDiv.hDiv ((algebraMap (Polyno …
  -/
  simp_rw [laurent_div, taylor_taylor]
  /-
    🎉 no goals
  -/


theorem laurent_injective : Function.Injective (laurent r) := fun _ _ h => by
  /-
    R : Type u
    inst✝¹ : CommRing R
    r : R
    inst✝ : IsDomain R
    x✝¹ x✝ : RatFunc R
    h : Eq ((RatFunc.laurent r) x✝¹) ((RatFunc.laurent r) x✝)
    ⊢ Eq x✝¹ x✝
  -/
  simpa [laurent_laurent] using congr_arg (laurent (-r)) h
  /-
    🎉 no goals
  -/


