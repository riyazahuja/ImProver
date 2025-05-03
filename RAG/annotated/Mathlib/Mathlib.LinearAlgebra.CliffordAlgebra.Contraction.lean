/-- Auxiliary construction for `CliffordAlgebra.contractLeft` -/
@[simps!]
def contractLeftAux (d : Module.Dual R M) :
    M →ₗ[R] CliffordAlgebra Q × CliffordAlgebra Q →ₗ[R] CliffordAlgebra Q :=
  haveI v_mul := (Algebra.lmul R (CliffordAlgebra Q)).toLinearMap ∘ₗ ι Q
  d.smulRight (LinearMap.fst _ (CliffordAlgebra Q) (CliffordAlgebra Q)) -
    v_mul.compl₂ (LinearMap.snd _ (CliffordAlgebra Q) _)


theorem contractLeftAux_contractLeftAux (v : M) (x : CliffordAlgebra Q) (fx : CliffordAlgebra Q) :
    contractLeftAux Q d v (ι Q v * x, contractLeftAux Q d v (x, fx)) = Q v • fx := by
  /-
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    d : Module.Dual R M
    v : M
    x fx : CliffordAlgebra Q
    ⊢ Eq (((CliffordAlgebra.contractLeftAux Q d) v) { fst := HMul.hMul ((CliffordA …
  -/
  simp only [contractLeftAux_apply_apply]
  rw [mul_sub, ← mul_assoc, ι_sq_scalar, ← Algebra.smul_def, ← sub_add, mul_smul_comm, sub_self,
    zero_add]


/-- Contract an element of the clifford algebra with an element `d : Module.Dual R M` from the left.

Note that $v ⌋ x$ is spelt `contractLeft (Q.associated v) x`.

This includes [grinberg_clifford_2016][] Theorem 10.75 -/
def contractLeft : Module.Dual R M →ₗ[R] CliffordAlgebra Q →ₗ[R] CliffordAlgebra Q where
  toFun d := foldr' Q (contractLeftAux Q d) (contractLeftAux_contractLeftAux Q d) 0
  map_add' d₁ d₂ :=
    LinearMap.ext fun x => by
      /-
        R : Type u1
        inst✝² : CommRing R
        M : Type u2
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        Q : QuadraticForm R M
        d d' d₁ d₂ : Module.Dual R M
        x : CliffordAlgebra Q
        ⊢ Eq (((fun d => CliffordAlgebra.foldr' Q (CliffordAlgebra.contractLeftAux Q d …
      -/
      dsimp only
      /-
        R : Type u1
        inst✝² : CommRing R
        M : Type u2
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        Q : QuadraticForm R M
        d d' d₁ d₂ : Module.Dual R M
        x : CliffordAlgebra Q
        ⊢ Eq ((CliffordAlgebra.foldr' Q (CliffordAlgebra.contractLeftAux Q (HAdd.hAdd  …
      -/
      rw [LinearMap.add_apply]
      /-
        R : Type u1
        inst✝² : CommRing R
        M : Type u2
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        Q : QuadraticForm R M
        d d' d₁ d₂ : Module.Dual R M
        x : CliffordAlgebra Q
        ⊢ Eq ((CliffordAlgebra.foldr' Q (CliffordAlgebra.contractLeftAux Q (HAdd.hAdd  …
      -/
      induction' x using CliffordAlgebra.left_induction with r x y hx hy m x hx
        /-
          case algebraMap
          R : Type u1
          inst✝² : CommRing R
          M : Type u2
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          Q : QuadraticForm R M
          d d' d₁ d₂ : Module.Dual R M
          r : R
          ⊢ Eq ((CliffordAlgebra.foldr' Q (CliffordAlgebra.contractLeftAux Q (HAdd.hAdd  …
        -/
      · simp_rw [foldr'_algebraMap, smul_zero, zero_add]
        /-
          🎉 no goals
        -/
        /-
          case add
          R : Type u1
          inst✝² : CommRing R
          M : Type u2
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          Q : QuadraticForm R M
          d d' d₁ d₂ : Module.Dual R M
          x y : CliffordAlgebra Q
          hx : Eq ((CliffordAlgebra.foldr' Q (CliffordAlgebra.contractLeftAux Q (HAdd.hA …
          hy : Eq ((CliffordAlgebra.foldr' Q (CliffordAlgebra.contractLeftAux Q (HAdd.hA …
          ⊢ Eq ((CliffordAlgebra.foldr' Q (CliffordAlgebra.contractLeftAux Q (HAdd.hAdd  …
        -/
      · rw [map_add, map_add, map_add, add_add_add_comm, hx, hy]
        /-
          🎉 no goals
        -/
        /-
          case ι_mul
          R : Type u1
          inst✝² : CommRing R
          M : Type u2
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          Q : QuadraticForm R M
          d d' d₁ d₂ : Module.Dual R M
          m : CliffordAlgebra Q
          x : M
          hx : Eq ((CliffordAlgebra.foldr' Q (CliffordAlgebra.contractLeftAux Q (HAdd.hA …
          ⊢ Eq ((CliffordAlgebra.foldr' Q (CliffordAlgebra.contractLeftAux Q (HAdd.hAdd  …
        -/
      · rw [foldr'_ι_mul, foldr'_ι_mul, foldr'_ι_mul, hx]
        /-
          case ι_mul
          R : Type u1
          inst✝² : CommRing R
          M : Type u2
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          Q : QuadraticForm R M
          d d' d₁ d₂ : Module.Dual R M
          m : CliffordAlgebra Q
          x : M
          hx : Eq ((CliffordAlgebra.foldr' Q (CliffordAlgebra.contractLeftAux Q (HAdd.hA …
          ⊢ Eq (((CliffordAlgebra.contractLeftAux Q (HAdd.hAdd d₁ d₂)) x) { fst := m, sn …
        -/
        dsimp only [contractLeftAux_apply_apply]
        /-
          case ι_mul
          R : Type u1
          inst✝² : CommRing R
          M : Type u2
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          Q : QuadraticForm R M
          d d' d₁ d₂ : Module.Dual R M
          m : CliffordAlgebra Q
          x : M
          hx : Eq ((CliffordAlgebra.foldr' Q (CliffordAlgebra.contractLeftAux Q (HAdd.hA …
          ⊢ Eq (HSub.hSub (HSMul.hSMul ((HAdd.hAdd d₁ d₂) x) m) (HMul.hMul ((CliffordAlg …
        -/
        rw [sub_add_sub_comm, mul_add, LinearMap.add_apply, add_smul]
        /-
          🎉 no goals
        -/
  map_smul' c d :=
    LinearMap.ext fun x => by
      /-
        R : Type u1
        inst✝² : CommRing R
        M : Type u2
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        Q : QuadraticForm R M
        d✝ d' : Module.Dual R M
        c : R
        d : Module.Dual R M
        x : CliffordAlgebra Q
        ⊢ Eq (({ toFun := fun d => CliffordAlgebra.foldr' Q (CliffordAlgebra.contractL …
      -/
      dsimp only
      /-
        R : Type u1
        inst✝² : CommRing R
        M : Type u2
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        Q : QuadraticForm R M
        d✝ d' : Module.Dual R M
        c : R
        d : Module.Dual R M
        x : CliffordAlgebra Q
        ⊢ Eq ((CliffordAlgebra.foldr' Q (CliffordAlgebra.contractLeftAux Q (HSMul.hSMu …
      -/
      rw [LinearMap.smul_apply, RingHom.id_apply]
      /-
        R : Type u1
        inst✝² : CommRing R
        M : Type u2
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        Q : QuadraticForm R M
        d✝ d' : Module.Dual R M
        c : R
        d : Module.Dual R M
        x : CliffordAlgebra Q
        ⊢ Eq ((CliffordAlgebra.foldr' Q (CliffordAlgebra.contractLeftAux Q (HSMul.hSMu …
      -/
      induction' x using CliffordAlgebra.left_induction with r x y hx hy m x hx
        /-
          case algebraMap
          R : Type u1
          inst✝² : CommRing R
          M : Type u2
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          Q : QuadraticForm R M
          d✝ d' : Module.Dual R M
          c : R
          d : Module.Dual R M
          r : R
          ⊢ Eq ((CliffordAlgebra.foldr' Q (CliffordAlgebra.contractLeftAux Q (HSMul.hSMu …
        -/
      · simp_rw [foldr'_algebraMap, smul_zero]
        /-
          🎉 no goals
        -/
        /-
          case add
          R : Type u1
          inst✝² : CommRing R
          M : Type u2
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          Q : QuadraticForm R M
          d✝ d' : Module.Dual R M
          c : R
          d : Module.Dual R M
          x y : CliffordAlgebra Q
          hx : Eq ((CliffordAlgebra.foldr' Q (CliffordAlgebra.contractLeftAux Q (HSMul.h …
          hy : Eq ((CliffordAlgebra.foldr' Q (CliffordAlgebra.contractLeftAux Q (HSMul.h …
          ⊢ Eq ((CliffordAlgebra.foldr' Q (CliffordAlgebra.contractLeftAux Q (HSMul.hSMu …
        -/
      · rw [map_add, map_add, smul_add, hx, hy]
        /-
          🎉 no goals
        -/
        /-
          case ι_mul
          R : Type u1
          inst✝² : CommRing R
          M : Type u2
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          Q : QuadraticForm R M
          d✝ d' : Module.Dual R M
          c : R
          d : Module.Dual R M
          m : CliffordAlgebra Q
          x : M
          hx : Eq ((CliffordAlgebra.foldr' Q (CliffordAlgebra.contractLeftAux Q (HSMul.h …
          ⊢ Eq ((CliffordAlgebra.foldr' Q (CliffordAlgebra.contractLeftAux Q (HSMul.hSMu …
        -/
      · rw [foldr'_ι_mul, foldr'_ι_mul, hx]
        /-
          case ι_mul
          R : Type u1
          inst✝² : CommRing R
          M : Type u2
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          Q : QuadraticForm R M
          d✝ d' : Module.Dual R M
          c : R
          d : Module.Dual R M
          m : CliffordAlgebra Q
          x : M
          hx : Eq ((CliffordAlgebra.foldr' Q (CliffordAlgebra.contractLeftAux Q (HSMul.h …
          ⊢ Eq (((CliffordAlgebra.contractLeftAux Q (HSMul.hSMul c d)) x) { fst := m, sn …
        -/
        dsimp only [contractLeftAux_apply_apply]
        /-
          case ι_mul
          R : Type u1
          inst✝² : CommRing R
          M : Type u2
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          Q : QuadraticForm R M
          d✝ d' : Module.Dual R M
          c : R
          d : Module.Dual R M
          m : CliffordAlgebra Q
          x : M
          hx : Eq ((CliffordAlgebra.foldr' Q (CliffordAlgebra.contractLeftAux Q (HSMul.h …
          ⊢ Eq (HSub.hSub (HSMul.hSMul ((HSMul.hSMul c d) x) m) (HMul.hMul ((CliffordAlg …
        -/
        rw [LinearMap.smul_apply, smul_assoc, mul_smul_comm, smul_sub]
        /-
          🎉 no goals
        -/


/-- Contract an element of the clifford algebra with an element `d : Module.Dual R M` from the
right.

Note that $x ⌊ v$ is spelt `contractRight x (Q.associated v)`.

This includes [grinberg_clifford_2016][] Theorem 16.75 -/
def contractRight : CliffordAlgebra Q →ₗ[R] Module.Dual R M →ₗ[R] CliffordAlgebra Q :=
  LinearMap.flip (LinearMap.compl₂ (LinearMap.compr₂ contractLeft reverse) reverse)


theorem contractRight_eq (x : CliffordAlgebra Q) :
    contractRight (Q := Q) x d = reverse (contractLeft (R := R) (M := M) d <| reverse x) :=
  rfl


local infixl:70 "⌋" => contractLeft (R := R) (M := M)


local infixl:70 "⌊" => contractRight (R := R) (M := M) (Q := Q)

-- Porting note: Lean needs to be reminded of this instance otherwise the statement of the
-- next result times out

instance : SMul R (CliffordAlgebra Q) := inferInstance


/-- This is [grinberg_clifford_2016][] Theorem 6  -/
theorem contractLeft_ι_mul (a : M) (b : CliffordAlgebra Q) :
    d⌋(ι Q a * b) = d a • b - ι Q a * (d⌋b) := by
-- Porting note: Lean cannot figure out anymore the third argument
  /-
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    d : Module.Dual R M
    a : M
    b : CliffordAlgebra Q
    ⊢ Eq ((CliffordAlgebra.contractLeft d) (HMul.hMul ((CliffordAlgebra.ι Q) a) b) …
  -/
  refine foldr'_ι_mul _ _ ?_ _ _ _
  /-
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    d : Module.Dual R M
    a : M
    b : CliffordAlgebra Q
    ⊢ ∀ (m : M) (x fx : CliffordAlgebra Q), Eq (((CliffordAlgebra.contractLeftAux  …
  -/
  exact fun m x fx ↦ contractLeftAux_contractLeftAux Q d m x fx
  /-
    🎉 no goals
  -/


/-- This is [grinberg_clifford_2016][] Theorem 12  -/
theorem contractRight_mul_ι (a : M) (b : CliffordAlgebra Q) :
    b * ι Q a⌊d = d a • b - b⌊d * ι Q a := by
  rw [contractRight_eq, reverse.map_mul, reverse_ι, contractLeft_ι_mul, map_sub, map_smul,
    reverse_reverse, reverse.map_mul, reverse_ι, contractRight_eq]


theorem contractLeft_algebraMap_mul (r : R) (b : CliffordAlgebra Q) :
    d⌋(algebraMap _ _ r * b) = algebraMap _ _ r * (d⌋b) := by
  /-
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    d : Module.Dual R M
    r : R
    b : CliffordAlgebra Q
    ⊢ Eq ((CliffordAlgebra.contractLeft d) (HMul.hMul ((algebraMap R (CliffordAlge …
  -/
  rw [← Algebra.smul_def, map_smul, Algebra.smul_def]
  /-
    🎉 no goals
  -/


theorem contractLeft_mul_algebraMap (a : CliffordAlgebra Q) (r : R) :
    d⌋(a * algebraMap _ _ r) = d⌋a * algebraMap _ _ r := by
  /-
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    d : Module.Dual R M
    a : CliffordAlgebra Q
    r : R
    ⊢ Eq ((CliffordAlgebra.contractLeft d) (HMul.hMul a ((algebraMap R (CliffordAl …
  -/
  rw [← Algebra.commutes, contractLeft_algebraMap_mul, Algebra.commutes]
  /-
    🎉 no goals
  -/


theorem contractRight_algebraMap_mul (r : R) (b : CliffordAlgebra Q) :
    algebraMap _ _ r * b⌊d = algebraMap _ _ r * (b⌊d) := by
  /-
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    d : Module.Dual R M
    r : R
    b : CliffordAlgebra Q
    ⊢ Eq ((CliffordAlgebra.contractRight (HMul.hMul ((algebraMap R (CliffordAlgebr …
  -/
  rw [← Algebra.smul_def, LinearMap.map_smul₂, Algebra.smul_def]
  /-
    🎉 no goals
  -/


theorem contractRight_mul_algebraMap (a : CliffordAlgebra Q) (r : R) :
    a * algebraMap _ _ r⌊d = a⌊d * algebraMap _ _ r := by
  /-
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    d : Module.Dual R M
    a : CliffordAlgebra Q
    r : R
    ⊢ Eq ((CliffordAlgebra.contractRight (HMul.hMul a ((algebraMap R (CliffordAlge …
  -/
  rw [← Algebra.commutes, contractRight_algebraMap_mul, Algebra.commutes]
  /-
    🎉 no goals
  -/


@[simp]
theorem contractLeft_ι (x : M) : d⌋ι Q x = algebraMap R _ (d x) := by
-- Porting note: Lean cannot figure out anymore the third argument
  refine (foldr'_ι _ _ ?_ _ _).trans <| by
    simp_rw [contractLeftAux_apply_apply, mul_zero, sub_zero,
      Algebra.algebraMap_eq_smul_one]
  /-
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    d : Module.Dual R M
    x : M
    ⊢ ∀ (m : M) (x fx : CliffordAlgebra Q), Eq (((CliffordAlgebra.contractLeftAux  …
  -/
  exact fun m x fx ↦ contractLeftAux_contractLeftAux Q d m x fx
  /-
    🎉 no goals
  -/


@[simp]
theorem contractRight_ι (x : M) : ι Q x⌊d = algebraMap R _ (d x) := by
  /-
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    d : Module.Dual R M
    x : M
    ⊢ Eq ((CliffordAlgebra.contractRight ((CliffordAlgebra.ι Q) x)) d) ((algebraMa …
  -/
  rw [contractRight_eq, reverse_ι, contractLeft_ι, reverse.commutes]
  /-
    🎉 no goals
  -/


@[simp]
theorem contractLeft_algebraMap (r : R) : d⌋algebraMap R (CliffordAlgebra Q) r = 0 := by
-- Porting note: Lean cannot figure out anymore the third argument
  /-
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    d : Module.Dual R M
    r : R
    ⊢ Eq ((CliffordAlgebra.contractLeft d) ((algebraMap R (CliffordAlgebra Q)) r)) 0
  -/
  refine (foldr'_algebraMap _ _ ?_ _ _).trans <| smul_zero _
  /-
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    d : Module.Dual R M
    r : R
    ⊢ ∀ (m : M) (x fx : CliffordAlgebra Q), Eq (((CliffordAlgebra.contractLeftAux  …
  -/
  exact fun m x fx ↦ contractLeftAux_contractLeftAux Q d m x fx
  /-
    🎉 no goals
  -/


@[simp]
theorem contractRight_algebraMap (r : R) : algebraMap R (CliffordAlgebra Q) r⌊d = 0 := by
  /-
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    d : Module.Dual R M
    r : R
    ⊢ Eq ((CliffordAlgebra.contractRight ((algebraMap R (CliffordAlgebra Q)) r)) d …
  -/
  rw [contractRight_eq, reverse.commutes, contractLeft_algebraMap, map_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem contractLeft_one : d⌋(1 : CliffordAlgebra Q) = 0 := by
  /-
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    d : Module.Dual R M
    ⊢ Eq ((CliffordAlgebra.contractLeft d) 1) 0
  -/
  simpa only [map_one] using contractLeft_algebraMap Q d 1
  /-
    🎉 no goals
  -/


@[simp]
theorem contractRight_one : (1 : CliffordAlgebra Q)⌊d = 0 := by
  /-
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    d : Module.Dual R M
    ⊢ Eq ((CliffordAlgebra.contractRight 1) d) 0
  -/
  simpa only [map_one] using contractRight_algebraMap Q d 1
  /-
    🎉 no goals
  -/


/-- This is [grinberg_clifford_2016][] Theorem 7 -/
theorem contractLeft_contractLeft (x : CliffordAlgebra Q) : d⌋(d⌋x) = 0 := by
  /-
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    d : Module.Dual R M
    x : CliffordAlgebra Q
    ⊢ Eq ((CliffordAlgebra.contractLeft d) ((CliffordAlgebra.contractLeft d) x)) 0
  -/
  induction' x using CliffordAlgebra.left_induction with r x y hx hy m x hx
    /-
      case algebraMap
      R : Type u1
      inst✝² : CommRing R
      M : Type u2
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      Q : QuadraticForm R M
      d : Module.Dual R M
      r : R
      ⊢ Eq ((CliffordAlgebra.contractLeft d) ((CliffordAlgebra.contractLeft d) ((alg …
    -/
  · simp_rw [contractLeft_algebraMap, map_zero]
    /-
      🎉 no goals
    -/
    /-
      case add
      R : Type u1
      inst✝² : CommRing R
      M : Type u2
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      Q : QuadraticForm R M
      d : Module.Dual R M
      x y : CliffordAlgebra Q
      hx : Eq ((CliffordAlgebra.contractLeft d) ((CliffordAlgebra.contractLeft d) x) …
      hy : Eq ((CliffordAlgebra.contractLeft d) ((CliffordAlgebra.contractLeft d) y) …
      ⊢ Eq ((CliffordAlgebra.contractLeft d) ((CliffordAlgebra.contractLeft d) (HAdd …
    -/
  · rw [map_add, map_add, hx, hy, add_zero]
    /-
      🎉 no goals
    -/
  · rw [contractLeft_ι_mul, map_sub, contractLeft_ι_mul, hx, LinearMap.map_smul,
      mul_zero, sub_zero, sub_self]


/-- This is [grinberg_clifford_2016][] Theorem 13 -/
theorem contractRight_contractRight (x : CliffordAlgebra Q) : x⌊d⌊d = 0 := by
  /-
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    d : Module.Dual R M
    x : CliffordAlgebra Q
    ⊢ Eq ((CliffordAlgebra.contractRight ((CliffordAlgebra.contractRight x) d)) d) 0
  -/
  rw [contractRight_eq, contractRight_eq, reverse_reverse, contractLeft_contractLeft, map_zero]
  /-
    🎉 no goals
  -/


/-- This is [grinberg_clifford_2016][] Theorem 8 -/
theorem contractLeft_comm (x : CliffordAlgebra Q) : d⌋(d'⌋x) = -(d'⌋(d⌋x)) := by
  /-
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    d d' : Module.Dual R M
    x : CliffordAlgebra Q
    ⊢ Eq ((CliffordAlgebra.contractLeft d) ((CliffordAlgebra.contractLeft d') x))  …
  -/
  induction' x using CliffordAlgebra.left_induction with r x y hx hy m x hx
    /-
      case algebraMap
      R : Type u1
      inst✝² : CommRing R
      M : Type u2
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      Q : QuadraticForm R M
      d d' : Module.Dual R M
      r : R
      ⊢ Eq ((CliffordAlgebra.contractLeft d) ((CliffordAlgebra.contractLeft d') ((al …
    -/
  · simp_rw [contractLeft_algebraMap, map_zero, neg_zero]
    /-
      🎉 no goals
    -/
    /-
      case add
      R : Type u1
      inst✝² : CommRing R
      M : Type u2
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      Q : QuadraticForm R M
      d d' : Module.Dual R M
      x y : CliffordAlgebra Q
      hx : Eq ((CliffordAlgebra.contractLeft d) ((CliffordAlgebra.contractLeft d') x …
      hy : Eq ((CliffordAlgebra.contractLeft d) ((CliffordAlgebra.contractLeft d') y …
      ⊢ Eq ((CliffordAlgebra.contractLeft d) ((CliffordAlgebra.contractLeft d') (HAd …
    -/
  · rw [map_add, map_add, map_add, map_add, hx, hy, neg_add]
    /-
      🎉 no goals
    -/
    /-
      case ι_mul
      R : Type u1
      inst✝² : CommRing R
      M : Type u2
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      Q : QuadraticForm R M
      d d' : Module.Dual R M
      m : CliffordAlgebra Q
      x : M
      hx : Eq ((CliffordAlgebra.contractLeft d) ((CliffordAlgebra.contractLeft d') m …
      ⊢ Eq ((CliffordAlgebra.contractLeft d) ((CliffordAlgebra.contractLeft d') (HMu …
    -/
  · simp only [contractLeft_ι_mul, map_sub, LinearMap.map_smul]
    /-
      case ι_mul
      R : Type u1
      inst✝² : CommRing R
      M : Type u2
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      Q : QuadraticForm R M
      d d' : Module.Dual R M
      m : CliffordAlgebra Q
      x : M
      hx : Eq ((CliffordAlgebra.contractLeft d) ((CliffordAlgebra.contractLeft d') m …
      ⊢ Eq (HSub.hSub (HSMul.hSMul (d' x) ((CliffordAlgebra.contractLeft d) m)) (HSu …
    -/
    rw [neg_sub, sub_sub_eq_add_sub, hx, mul_neg, ← sub_eq_add_neg]
    /-
      🎉 no goals
    -/


/-- This is [grinberg_clifford_2016][] Theorem 14 -/
theorem contractRight_comm (x : CliffordAlgebra Q) : x⌊d⌊d' = -(x⌊d'⌊d) := by
  rw [contractRight_eq, contractRight_eq, contractRight_eq, contractRight_eq, reverse_reverse,
    reverse_reverse, contractLeft_comm, map_neg]

/- TODO:
lemma contractRight_contractLeft (x : CliffordAlgebra Q) : (d ⌋ x) ⌊ d' = d ⌋ (x ⌊ d') :=
-/

local infixl:70 "⌋" => contractLeft


local infixl:70 "⌊" => contractRight


/-- Auxiliary construction for `CliffordAlgebra.changeForm` -/
@[simps!]
def changeFormAux (B : BilinForm R M) : M →ₗ[R] CliffordAlgebra Q →ₗ[R] CliffordAlgebra Q :=
  haveI v_mul := (Algebra.lmul R (CliffordAlgebra Q)).toLinearMap ∘ₗ ι Q
  v_mul - contractLeft ∘ₗ B


theorem changeFormAux_changeFormAux (B : BilinForm R M) (v : M) (x : CliffordAlgebra Q) :
    changeFormAux Q B v (changeFormAux Q B v x) = (Q v - B v v) • x := by
  /-
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    B : LinearMap.BilinForm R M
    v : M
    x : CliffordAlgebra Q
    ⊢ Eq (((CliffordAlgebra.changeFormAux Q B) v) (((CliffordAlgebra.changeFormAux …
  -/
  simp only [changeFormAux_apply_apply]
  rw [mul_sub, ← mul_assoc, ι_sq_scalar, map_sub, contractLeft_ι_mul, ← sub_add, sub_sub_sub_comm,
    ← Algebra.smul_def, sub_self, sub_zero, contractLeft_contractLeft, add_zero, sub_smul]


/-- Convert between two algebras of different quadratic form, sending vector to vectors, scalars to
scalars, and adjusting products by a contraction term.

This is $\lambda_B$ from [bourbaki2007][] $9 Lemma 2. -/
def changeForm (h : B.toQuadraticMap = Q' - Q) : CliffordAlgebra Q →ₗ[R] CliffordAlgebra Q' :=
  foldr Q (changeFormAux Q' B)
    (fun m x =>
      (changeFormAux_changeFormAux Q' B m x).trans <| by
        /-
          R : Type u1
          inst✝² : CommRing R
          M : Type u2
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          Q Q' Q'' : QuadraticForm R M
          B B' : LinearMap.BilinForm R M
          h : Eq (LinearMap.BilinMap.toQuadraticMap B) (HSub.hSub Q' Q)
          m : M
          x : CliffordAlgebra Q'
          ⊢ Eq (HSMul.hSMul (HSub.hSub (Q' m) ((B m) m)) x) (HSMul.hSMul (Q m) x)
        -/
        dsimp only [← BilinMap.toQuadraticMap_apply]
        /-
          R : Type u1
          inst✝² : CommRing R
          M : Type u2
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          Q Q' Q'' : QuadraticForm R M
          B B' : LinearMap.BilinForm R M
          h : Eq (LinearMap.BilinMap.toQuadraticMap B) (HSub.hSub Q' Q)
          m : M
          x : CliffordAlgebra Q'
          ⊢ Eq (HSMul.hSMul (HSub.hSub (Q' m) ((LinearMap.BilinMap.toQuadraticMap B) m)) …
        -/
        rw [h, QuadraticMap.sub_apply, sub_sub_cancel])
        /-
          🎉 no goals
        -/
    1


/-- Auxiliary lemma used as an argument to `CliffordAlgebra.changeForm` -/
theorem changeForm.zero_proof : (0 : BilinForm R M).toQuadraticMap = Q - Q :=
  (sub_self _).symm


include h h' in
/-- Auxiliary lemma used as an argument to `CliffordAlgebra.changeForm` -/
theorem changeForm.add_proof : (B + B').toQuadraticMap = Q'' - Q :=
  (congr_arg₂ (· + ·) h h').trans <| sub_add_sub_cancel' _ _ _


include h in
/-- Auxiliary lemma used as an argument to `CliffordAlgebra.changeForm` -/
theorem changeForm.neg_proof : (-B).toQuadraticMap = Q - Q' :=
  (congr_arg Neg.neg h).trans <| neg_sub _ _


theorem changeForm.associated_neg_proof [Invertible (2 : R)] :
    (QuadraticMap.associated (R := R) (M := M) (-Q)).toQuadraticMap = 0 - Q := by
  /-
    R : Type u1
    inst✝³ : CommRing R
    M : Type u2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    Q : QuadraticForm R M
    inst✝ : Invertible 2
    ⊢ Eq (QuadraticMap.associated (Neg.neg Q)).toQuadraticMap (HSub.hSub 0 Q)
  -/
  simp [QuadraticMap.toQuadraticMap_associated]
  /-
    🎉 no goals
  -/


@[simp]
theorem changeForm_algebraMap (r : R) : changeForm h (algebraMap R _ r) = algebraMap R _ r :=
  (foldr_algebraMap _ _ _ _ _).trans <| Eq.symm <| Algebra.algebraMap_eq_smul_one r


@[simp]
theorem changeForm_one : changeForm h (1 : CliffordAlgebra Q) = 1 := by
  /-
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q Q' : QuadraticForm R M
    B : LinearMap.BilinForm R M
    h : Eq (LinearMap.BilinMap.toQuadraticMap B) (HSub.hSub Q' Q)
    ⊢ Eq ((CliffordAlgebra.changeForm h) 1) 1
  -/
  simpa using changeForm_algebraMap h (1 : R)
  /-
    🎉 no goals
  -/


@[simp]
theorem changeForm_ι (m : M) : changeForm h (ι (M := M) Q m) = ι (M := M) Q' m :=
  (foldr_ι _ _ _ _ _).trans <|
                  /-
                    R : Type u1
                    inst✝² : CommRing R
                    M : Type u2
                    inst✝¹ : AddCommGroup M
                    inst✝ : Module R M
                    Q Q' : QuadraticForm R M
                    B : LinearMap.BilinForm R M
                    h : Eq (LinearMap.BilinMap.toQuadraticMap B) (HSub.hSub Q' Q)
                    m : M
                    ⊢ Eq ((CliffordAlgebra.ι Q') m) (((CliffordAlgebra.changeFormAux Q' B) m) 1)
                  -/
    Eq.symm <| by rw [changeFormAux_apply_apply, mul_one, contractLeft_one, sub_zero]
                  /-
                    🎉 no goals
                  -/


theorem changeForm_ι_mul (m : M) (x : CliffordAlgebra Q) :
    changeForm h (ι (M := M) Q m * x) = ι (M := M) Q' m * changeForm h x
    - contractLeft (Q := Q') (B m) (changeForm h x) :=
-- Porting note: original statement
--    - BilinForm.toLin B m⌋changeForm h x :=
                                      /-
                                        R : Type u1
                                        inst✝² : CommRing R
                                        M : Type u2
                                        inst✝¹ : AddCommGroup M
                                        inst✝ : Module R M
                                        Q Q' : QuadraticForm R M
                                        B : LinearMap.BilinForm R M
                                        h : Eq (LinearMap.BilinMap.toQuadraticMap B) (HSub.hSub Q' Q)
                                        m : M
                                        x : CliffordAlgebra Q
                                        ⊢ Eq (((CliffordAlgebra.foldr Q (CliffordAlgebra.changeFormAux Q' B) ⋯) (((Cli …
                                      -/
  (foldr_mul _ _ _ _ _ _).trans <| by rw [foldr_ι]; rfl
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem changeForm_ι_mul_ι (m₁ m₂ : M) :
    changeForm h (ι Q m₁ * ι Q m₂) = ι Q' m₁ * ι Q' m₂ - algebraMap _ _ (B m₁ m₂) := by
  /-
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q Q' : QuadraticForm R M
    B : LinearMap.BilinForm R M
    h : Eq (LinearMap.BilinMap.toQuadraticMap B) (HSub.hSub Q' Q)
    m₁ m₂ : M
    ⊢ Eq ((CliffordAlgebra.changeForm h) (HMul.hMul ((CliffordAlgebra.ι Q) m₁) ((C …
  -/
  rw [changeForm_ι_mul, changeForm_ι, contractLeft_ι]
  /-
    🎉 no goals
  -/


/-- Theorem 23 of [grinberg_clifford_2016][] -/
theorem changeForm_contractLeft (d : Module.Dual R M) (x : CliffordAlgebra Q) :
    -- Porting note: original statement
    --    changeForm h (d⌋x) = d⌋changeForm h x := by
    changeForm h (contractLeft (Q := Q) d x) = contractLeft (Q := Q') d (changeForm h x) := by
  /-
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q Q' : QuadraticForm R M
    B : LinearMap.BilinForm R M
    h : Eq (LinearMap.BilinMap.toQuadraticMap B) (HSub.hSub Q' Q)
    d : Module.Dual R M
    x : CliffordAlgebra Q
    ⊢ Eq ((CliffordAlgebra.changeForm h) ((CliffordAlgebra.contractLeft d) x)) ((C …
  -/
  induction' x using CliffordAlgebra.left_induction with r x y hx hy m x hx
    /-
      case algebraMap
      R : Type u1
      inst✝² : CommRing R
      M : Type u2
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      Q Q' : QuadraticForm R M
      B : LinearMap.BilinForm R M
      h : Eq (LinearMap.BilinMap.toQuadraticMap B) (HSub.hSub Q' Q)
      d : Module.Dual R M
      r : R
      ⊢ Eq ((CliffordAlgebra.changeForm h) ((CliffordAlgebra.contractLeft d) ((algeb …
    -/
  · simp only [contractLeft_algebraMap, changeForm_algebraMap, map_zero]
    /-
      🎉 no goals
    -/
    /-
      case add
      R : Type u1
      inst✝² : CommRing R
      M : Type u2
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      Q Q' : QuadraticForm R M
      B : LinearMap.BilinForm R M
      h : Eq (LinearMap.BilinMap.toQuadraticMap B) (HSub.hSub Q' Q)
      d : Module.Dual R M
      x y : CliffordAlgebra Q
      hx : Eq ((CliffordAlgebra.changeForm h) ((CliffordAlgebra.contractLeft d) x))  …
      hy : Eq ((CliffordAlgebra.changeForm h) ((CliffordAlgebra.contractLeft d) y))  …
      ⊢ Eq ((CliffordAlgebra.changeForm h) ((CliffordAlgebra.contractLeft d) (HAdd.h …
    -/
  · rw [map_add, map_add, map_add, map_add, hx, hy]
    /-
      🎉 no goals
    -/
    /-
      case ι_mul
      R : Type u1
      inst✝² : CommRing R
      M : Type u2
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      Q Q' : QuadraticForm R M
      B : LinearMap.BilinForm R M
      h : Eq (LinearMap.BilinMap.toQuadraticMap B) (HSub.hSub Q' Q)
      d : Module.Dual R M
      m : CliffordAlgebra Q
      x : M
      hx : Eq ((CliffordAlgebra.changeForm h) ((CliffordAlgebra.contractLeft d) m))  …
      ⊢ Eq ((CliffordAlgebra.changeForm h) ((CliffordAlgebra.contractLeft d) (HMul.h …
    -/
  · simp only [contractLeft_ι_mul, changeForm_ι_mul, map_sub, LinearMap.map_smul]
    /-
      case ι_mul
      R : Type u1
      inst✝² : CommRing R
      M : Type u2
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      Q Q' : QuadraticForm R M
      B : LinearMap.BilinForm R M
      h : Eq (LinearMap.BilinMap.toQuadraticMap B) (HSub.hSub Q' Q)
      d : Module.Dual R M
      m : CliffordAlgebra Q
      x : M
      hx : Eq ((CliffordAlgebra.changeForm h) ((CliffordAlgebra.contractLeft d) m))  …
      ⊢ Eq (HSub.hSub (HSMul.hSMul (d x) ((CliffordAlgebra.changeForm h) m)) (HSub.h …
    -/
    rw [← hx, contractLeft_comm, ← sub_add, sub_neg_eq_add, ← hx]
    /-
      🎉 no goals
    -/


theorem changeForm_self_apply (x : CliffordAlgebra Q) : changeForm (Q' := Q)
    changeForm.zero_proof x = x := by
  /-
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    x : CliffordAlgebra Q
    ⊢ Eq ((CliffordAlgebra.changeForm ⋯) x) x
  -/
  induction' x using CliffordAlgebra.left_induction with r x y hx hy m x hx
    /-
      case algebraMap
      R : Type u1
      inst✝² : CommRing R
      M : Type u2
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      Q : QuadraticForm R M
      r : R
      ⊢ Eq ((CliffordAlgebra.changeForm ⋯) ((algebraMap R (CliffordAlgebra Q)) r)) ( …
    -/
  · simp_rw [changeForm_algebraMap]
    /-
      🎉 no goals
    -/
    /-
      case add
      R : Type u1
      inst✝² : CommRing R
      M : Type u2
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      Q : QuadraticForm R M
      x y : CliffordAlgebra Q
      hx : Eq ((CliffordAlgebra.changeForm ⋯) x) x
      hy : Eq ((CliffordAlgebra.changeForm ⋯) y) y
      ⊢ Eq ((CliffordAlgebra.changeForm ⋯) (HAdd.hAdd x y)) (HAdd.hAdd x y)
    -/
  · rw [map_add, hx, hy]
    /-
      🎉 no goals
    -/
  · rw [changeForm_ι_mul, hx, LinearMap.zero_apply, map_zero, LinearMap.zero_apply,
      sub_zero]


@[simp]
theorem changeForm_self :
    changeForm changeForm.zero_proof = (LinearMap.id : CliffordAlgebra Q →ₗ[R] _) :=
  LinearMap.ext <| changeForm_self_apply


/-- This is [bourbaki2007][] $9 Lemma 3. -/
theorem changeForm_changeForm (x : CliffordAlgebra Q) :
    changeForm h' (changeForm h x) = changeForm (changeForm.add_proof h h') x := by
  /-
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q Q' Q'' : QuadraticForm R M
    B B' : LinearMap.BilinForm R M
    h : Eq (LinearMap.BilinMap.toQuadraticMap B) (HSub.hSub Q' Q)
    h' : Eq (LinearMap.BilinMap.toQuadraticMap B') (HSub.hSub Q'' Q')
    x : CliffordAlgebra Q
    ⊢ Eq ((CliffordAlgebra.changeForm h') ((CliffordAlgebra.changeForm h) x)) ((Cl …
  -/
  induction' x using CliffordAlgebra.left_induction with r x y hx hy m x hx
    /-
      case algebraMap
      R : Type u1
      inst✝² : CommRing R
      M : Type u2
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      Q Q' Q'' : QuadraticForm R M
      B B' : LinearMap.BilinForm R M
      h : Eq (LinearMap.BilinMap.toQuadraticMap B) (HSub.hSub Q' Q)
      h' : Eq (LinearMap.BilinMap.toQuadraticMap B') (HSub.hSub Q'' Q')
      r : R
      ⊢ Eq ((CliffordAlgebra.changeForm h') ((CliffordAlgebra.changeForm h) ((algebr …
    -/
  · simp_rw [changeForm_algebraMap]
    /-
      🎉 no goals
    -/
    /-
      case add
      R : Type u1
      inst✝² : CommRing R
      M : Type u2
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      Q Q' Q'' : QuadraticForm R M
      B B' : LinearMap.BilinForm R M
      h : Eq (LinearMap.BilinMap.toQuadraticMap B) (HSub.hSub Q' Q)
      h' : Eq (LinearMap.BilinMap.toQuadraticMap B') (HSub.hSub Q'' Q')
      x y : CliffordAlgebra Q
      hx : Eq ((CliffordAlgebra.changeForm h') ((CliffordAlgebra.changeForm h) x)) ( …
      hy : Eq ((CliffordAlgebra.changeForm h') ((CliffordAlgebra.changeForm h) y)) ( …
      ⊢ Eq ((CliffordAlgebra.changeForm h') ((CliffordAlgebra.changeForm h) (HAdd.hA …
    -/
  · rw [map_add, map_add, map_add, hx, hy]
    /-
      🎉 no goals
    -/
  · rw [changeForm_ι_mul, map_sub, changeForm_ι_mul, changeForm_ι_mul, hx, sub_sub,
      LinearMap.add_apply, map_add, LinearMap.add_apply, changeForm_contractLeft, hx,
      add_comm (_ : CliffordAlgebra Q'')]


theorem changeForm_comp_changeForm :
    (changeForm h').comp (changeForm h) = changeForm (changeForm.add_proof h h') :=
  LinearMap.ext <| changeForm_changeForm _ h'


/-- Any two algebras whose quadratic forms differ by a bilinear form are isomorphic as modules.

This is $\bar \lambda_B$ from [bourbaki2007][] $9 Proposition 3. -/
@[simps apply]
def changeFormEquiv : CliffordAlgebra Q ≃ₗ[R] CliffordAlgebra Q' :=
  { changeForm h with
    toFun := changeForm h
    invFun := changeForm (changeForm.neg_proof h)
    left_inv := fun x => by
      /-
        R : Type u1
        inst✝² : CommRing R
        M : Type u2
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        Q Q' Q'' : QuadraticForm R M
        B B' : LinearMap.BilinForm R M
        h : Eq (LinearMap.BilinMap.toQuadraticMap B) (HSub.hSub Q' Q)
        h' : Eq (LinearMap.BilinMap.toQuadraticMap B') (HSub.hSub Q'' Q')
        x : CliffordAlgebra Q
        ⊢ Eq ((CliffordAlgebra.changeForm ⋯) ({ toFun := ⇑(CliffordAlgebra.changeForm  …
      -/
      dsimp only
      exact (changeForm_changeForm _ _ x).trans <|
        by simp_rw [(add_neg_cancel B), changeForm_self_apply]
    right_inv := fun x => by
      /-
        R : Type u1
        inst✝² : CommRing R
        M : Type u2
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        Q Q' Q'' : QuadraticForm R M
        B B' : LinearMap.BilinForm R M
        h : Eq (LinearMap.BilinMap.toQuadraticMap B) (HSub.hSub Q' Q)
        h' : Eq (LinearMap.BilinMap.toQuadraticMap B') (HSub.hSub Q'' Q')
        x : CliffordAlgebra Q'
        ⊢ Eq ({ toFun := ⇑(CliffordAlgebra.changeForm h), map_add' := ⋯, map_smul' :=  …
      -/
      dsimp only
      exact (changeForm_changeForm _ _ x).trans <|
        by simp_rw [(neg_add_cancel B), changeForm_self_apply] }


@[simp]
theorem changeFormEquiv_symm :
    (changeFormEquiv h).symm = changeFormEquiv (changeForm.neg_proof h) :=
  LinearEquiv.ext fun _ => rfl


/-- The module isomorphism to the exterior algebra.

Note that this holds more generally when `Q` is divisible by two, rather than only when `1` is
divisible by two; but that would be more awkward to use. -/
@[simp]
def equivExterior [Invertible (2 : R)] : CliffordAlgebra Q ≃ₗ[R] ExteriorAlgebra R M :=
  changeFormEquiv changeForm.associated_neg_proof


/-- A `CliffordAlgebra` over a nontrivial ring is nontrivial, in characteristic not two. -/
instance [Nontrivial R] [Invertible (2 : R)] :
    Nontrivial (CliffordAlgebra Q) := (equivExterior Q).symm.injective.nontrivial


