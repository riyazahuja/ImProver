instance AlternatingMap.instModuleAddCommGroup {ι : Type*} :
    Module R (M [⋀^ι]→ₗ[R] N) := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    N' : Type u_4
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : AddCommGroup N'
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module R N'
    ι : Type u_5
    ⊢ Module R (AlternatingMap R M N ι)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- Build a map out of the exterior algebra given a collection of alternating maps acting on each
exterior power -/
def liftAlternating : (∀ i, M [⋀^Fin i]→ₗ[R] N) →ₗ[R] ExteriorAlgebra R M →ₗ[R] N := by
  suffices
    (∀ i, M [⋀^Fin i]→ₗ[R] N) →ₗ[R]
      ExteriorAlgebra R M →ₗ[R] ∀ i, M [⋀^Fin i]→ₗ[R] N by
    refine LinearMap.compr₂ this ?_
    refine (LinearEquiv.toLinearMap ?_).comp (LinearMap.proj 0)
    exact AlternatingMap.constLinearEquivOfIsEmpty.symm
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    N' : Type u_4
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : AddCommGroup N'
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module R N'
    ⊢ LinearMap (RingHom.id R) ((i : Nat) → AlternatingMap R M N (Fin i)) (LinearM …
  -/
  refine CliffordAlgebra.foldl _ ?_ ?_
  · refine
      LinearMap.mk₂ R (fun m f i => (f i.succ).curryLeft m) (fun m₁ m₂ f => ?_) (fun c m f => ?_)
        (fun m f₁ f₂ => ?_) fun c m f => ?_
    all_goals
      ext i : 1
      simp only [map_smul, map_add, Pi.add_apply, Pi.smul_apply, AlternatingMap.curryLeft_add,
        AlternatingMap.curryLeft_smul, map_add, map_smul, LinearMap.add_apply, LinearMap.smul_apply]
  · -- when applied twice with the same `m`, this recursive step produces 0
    /-
      case refine_2
      R : Type u_1
      M : Type u_2
      N : Type u_3
      N' : Type u_4
      inst✝⁶ : CommRing R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : AddCommGroup N
      inst✝³ : AddCommGroup N'
      inst✝² : Module R M
      inst✝¹ : Module R N
      inst✝ : Module R N'
      ⊢ ∀ (m : M) (x : (i : Nat) → AlternatingMap R M N (Fin i)), Eq (((LinearMap.mk …
    -/
    intro m x
    /-
      case refine_2
      R : Type u_1
      M : Type u_2
      N : Type u_3
      N' : Type u_4
      inst✝⁶ : CommRing R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : AddCommGroup N
      inst✝³ : AddCommGroup N'
      inst✝² : Module R M
      inst✝¹ : Module R N
      inst✝ : Module R N'
      m : M
      x : (i : Nat) → AlternatingMap R M N (Fin i)
      ⊢ Eq (((LinearMap.mk₂ R (fun m f i => (f i.succ).curryLeft m) ⋯ ⋯ ⋯ ⋯) m) (((L …
    -/
    ext
    /-
      case refine_2.h.H
      R : Type u_1
      M : Type u_2
      N : Type u_3
      N' : Type u_4
      inst✝⁶ : CommRing R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : AddCommGroup N
      inst✝³ : AddCommGroup N'
      inst✝² : Module R M
      inst✝¹ : Module R N
      inst✝ : Module R N'
      m : M
      x : (i : Nat) → AlternatingMap R M N (Fin i)
      x✝¹ : Nat
      x✝ : Fin x✝¹ → M
      ⊢ Eq ((((LinearMap.mk₂ R (fun m f i => (f i.succ).curryLeft m) ⋯ ⋯ ⋯ ⋯) m) ((( …
    -/
    simp
    /-
      🎉 no goals
    -/


@[simp]
theorem liftAlternating_ι (f : ∀ i, M [⋀^Fin i]→ₗ[R] N) (m : M) :
    liftAlternating (R := R) (M := M) (N := N) f (ι R m) = f 1 ![m] := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    f : (i : Nat) → AlternatingMap R M N (Fin i)
    m : M
    ⊢ Eq ((ExteriorAlgebra.liftAlternating f) ((ExteriorAlgebra.ι R) m)) ((f 1) (M …
  -/
  dsimp [liftAlternating]
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    f : (i : Nat) → AlternatingMap R M N (Fin i)
    m : M
    ⊢ Eq ((((CliffordAlgebra.foldl 0 (LinearMap.mk₂ R (fun m f i => (f (HAdd.hAdd  …
  -/
  rw [foldl_ι, LinearMap.mk₂_apply, AlternatingMap.curryLeft_apply_apply]
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    f : (i : Nat) → AlternatingMap R M N (Fin i)
    m : M
    ⊢ Eq ((f (HAdd.hAdd 0 1)) (Matrix.vecCons m 0)) ((f 1) (Matrix.vecCons m Matri …
  -/
  congr
  -- Porting note: In Lean 3, `congr` could use the `[Subsingleton (Fin 0 → M)]` instance to finish
  -- the proof. Here, the instance can be synthesized but `congr` does not use it so the following
  -- line is provided.
  /-
    case h.e_6.h
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    f : (i : Nat) → AlternatingMap R M N (Fin i)
    m : M
    ⊢ Eq (Matrix.vecCons m 0) (Matrix.vecCons m Matrix.vecEmpty)
  -/
  rw [Matrix.zero_empty]
  /-
    🎉 no goals
  -/


theorem liftAlternating_ι_mul (f : ∀ i, M [⋀^Fin i]→ₗ[R] N) (m : M)
    (x : ExteriorAlgebra R M) :
    liftAlternating (R := R) (M := M) (N := N) f (ι R m * x) =
    liftAlternating (R := R) (M := M) (N := N) (fun i => (f i.succ).curryLeft m) x := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    f : (i : Nat) → AlternatingMap R M N (Fin i)
    m : M
    x : ExteriorAlgebra R M
    ⊢ Eq ((ExteriorAlgebra.liftAlternating f) (HMul.hMul ((ExteriorAlgebra.ι R) m) …
  -/
  dsimp [liftAlternating]
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    f : (i : Nat) → AlternatingMap R M N (Fin i)
    m : M
    x : ExteriorAlgebra R M
    ⊢ Eq ((((CliffordAlgebra.foldl 0 (LinearMap.mk₂ R (fun m f i => (f (HAdd.hAdd  …
  -/
  rw [foldl_mul, foldl_ι]
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    f : (i : Nat) → AlternatingMap R M N (Fin i)
    m : M
    x : ExteriorAlgebra R M
    ⊢ Eq ((((CliffordAlgebra.foldl 0 (LinearMap.mk₂ R (fun m f i => (f (HAdd.hAdd  …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem liftAlternating_one (f : ∀ i, M [⋀^Fin i]→ₗ[R] N) :
    liftAlternating (R := R) (M := M) (N := N) f (1 : ExteriorAlgebra R M) = f 0 0 := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    f : (i : Nat) → AlternatingMap R M N (Fin i)
    ⊢ Eq ((ExteriorAlgebra.liftAlternating f) 1) ((f 0) 0)
  -/
  dsimp [liftAlternating]
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    f : (i : Nat) → AlternatingMap R M N (Fin i)
    ⊢ Eq ((((CliffordAlgebra.foldl 0 (LinearMap.mk₂ R (fun m f i => (f (HAdd.hAdd  …
  -/
  rw [foldl_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem liftAlternating_algebraMap (f : ∀ i, M [⋀^Fin i]→ₗ[R] N) (r : R) :
    liftAlternating (R := R) (M := M) (N := N) f (algebraMap _ (ExteriorAlgebra R M) r) =
    r • f 0 0 := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    f : (i : Nat) → AlternatingMap R M N (Fin i)
    r : R
    ⊢ Eq ((ExteriorAlgebra.liftAlternating f) ((algebraMap R (ExteriorAlgebra R M) …
  -/
  rw [Algebra.algebraMap_eq_smul_one, map_smul, liftAlternating_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem liftAlternating_apply_ιMulti {n : ℕ} (f : ∀ i, M [⋀^Fin i]→ₗ[R] N)
    (v : Fin n → M) : liftAlternating (R := R) (M := M) (N := N) f (ιMulti R n v) = f n v := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    n : Nat
    f : (i : Nat) → AlternatingMap R M N (Fin i)
    v : Fin n → M
    ⊢ Eq ((ExteriorAlgebra.liftAlternating f) ((ExteriorAlgebra.ιMulti R n) v)) (( …
  -/
  rw [ιMulti_apply]
  -- Porting note: `v` is generalized automatically so it was removed from the next line
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    n : Nat
    f : (i : Nat) → AlternatingMap R M N (Fin i)
    v : Fin n → M
    ⊢ Eq ((ExteriorAlgebra.liftAlternating f) (List.ofFn fun i => (ExteriorAlgebra …
  -/
  induction' n with n ih generalizing f
  · -- Porting note: Lean does not automatically synthesize the instance
    -- `[Subsingleton (Fin 0 → M)]` which is needed for `Subsingleton.elim 0 v` on line 114.
    /-
      case zero
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : AddCommGroup N
      inst✝¹ : Module R M
      inst✝ : Module R N
      f : (i : Nat) → AlternatingMap R M N (Fin i)
      v : Fin 0 → M
      ⊢ Eq ((ExteriorAlgebra.liftAlternating f) (List.ofFn fun i => (ExteriorAlgebra …
    -/
    letI : Subsingleton (Fin 0 → M) := by infer_instance
    /-
      case zero
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : AddCommGroup N
      inst✝¹ : Module R M
      inst✝ : Module R N
      f : (i : Nat) → AlternatingMap R M N (Fin i)
      v : Fin 0 → M
      this : Subsingleton (Fin 0 → M) := inferInstance
      ⊢ Eq ((ExteriorAlgebra.liftAlternating f) (List.ofFn fun i => (ExteriorAlgebra …
    -/
    rw [List.ofFn_zero, List.prod_nil, liftAlternating_one, Subsingleton.elim 0 v]
    /-
      🎉 no goals
    -/
  · rw [List.ofFn_succ, List.prod_cons, liftAlternating_ι_mul, ih,
      AlternatingMap.curryLeft_apply_apply]
    /-
      case succ
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : AddCommGroup N
      inst✝¹ : Module R M
      inst✝ : Module R N
      n : Nat
      ih : ∀ (f : (i : Nat) → AlternatingMap R M N (Fin i)) (v : Fin n → M), Eq ((Ex …
      f : (i : Nat) → AlternatingMap R M N (Fin i)
      v : Fin (HAdd.hAdd n 1) → M
      ⊢ Eq ((f n.succ) (Matrix.vecCons (v 0) fun i => v i.succ)) ((f (HAdd.hAdd n 1) …
    -/
    congr
    /-
      case succ.h.e_6.h
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : AddCommGroup N
      inst✝¹ : Module R M
      inst✝ : Module R N
      n : Nat
      ih : ∀ (f : (i : Nat) → AlternatingMap R M N (Fin i)) (v : Fin n → M), Eq ((Ex …
      f : (i : Nat) → AlternatingMap R M N (Fin i)
      v : Fin (HAdd.hAdd n 1) → M
      ⊢ Eq (Matrix.vecCons (v 0) fun i => v i.succ) v
    -/
    exact Matrix.cons_head_tail _
    /-
      🎉 no goals
    -/


@[simp]
theorem liftAlternating_comp_ιMulti {n : ℕ} (f : ∀ i, M [⋀^Fin i]→ₗ[R] N) :
    (liftAlternating (R := R) (M := M) (N := N) f).compAlternatingMap (ιMulti R n) = f n :=
  AlternatingMap.ext <| liftAlternating_apply_ιMulti f


@[simp]
theorem liftAlternating_comp (g : N →ₗ[R] N') (f : ∀ i, M [⋀^Fin i]→ₗ[R] N) :
    (liftAlternating (R := R) (M := M) (N := N') fun i => g.compAlternatingMap (f i)) =
    g ∘ₗ liftAlternating (R := R) (M := M) (N := N) f := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    N' : Type u_4
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : AddCommGroup N'
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module R N'
    g : LinearMap (RingHom.id R) N N'
    f : (i : Nat) → AlternatingMap R M N (Fin i)
    ⊢ Eq (ExteriorAlgebra.liftAlternating fun i => g.compAlternatingMap (f i)) (g. …
  -/
  ext v
  /-
    case h
    R : Type u_1
    M : Type u_2
    N : Type u_3
    N' : Type u_4
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : AddCommGroup N'
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module R N'
    g : LinearMap (RingHom.id R) N N'
    f : (i : Nat) → AlternatingMap R M N (Fin i)
    v : ExteriorAlgebra R M
    ⊢ Eq ((ExteriorAlgebra.liftAlternating fun i => g.compAlternatingMap (f i)) v) …
  -/
  rw [LinearMap.comp_apply]
  /-
    case h
    R : Type u_1
    M : Type u_2
    N : Type u_3
    N' : Type u_4
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : AddCommGroup N'
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module R N'
    g : LinearMap (RingHom.id R) N N'
    f : (i : Nat) → AlternatingMap R M N (Fin i)
    v : ExteriorAlgebra R M
    ⊢ Eq ((ExteriorAlgebra.liftAlternating fun i => g.compAlternatingMap (f i)) v) …
  -/
  induction' v using CliffordAlgebra.left_induction with r x y hx hy x m hx generalizing f
  · rw [liftAlternating_algebraMap, liftAlternating_algebraMap, map_smul,
      LinearMap.compAlternatingMap_apply]
    /-
      case h.add
      R : Type u_1
      M : Type u_2
      N : Type u_3
      N' : Type u_4
      inst✝⁶ : CommRing R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : AddCommGroup N
      inst✝³ : AddCommGroup N'
      inst✝² : Module R M
      inst✝¹ : Module R N
      inst✝ : Module R N'
      g : LinearMap (RingHom.id R) N N'
      x y : CliffordAlgebra 0
      hx : ∀ (f : (i : Nat) → AlternatingMap R M N (Fin i)), Eq ((ExteriorAlgebra.li …
      hy : ∀ (f : (i : Nat) → AlternatingMap R M N (Fin i)), Eq ((ExteriorAlgebra.li …
      f : (i : Nat) → AlternatingMap R M N (Fin i)
      ⊢ Eq ((ExteriorAlgebra.liftAlternating fun i => g.compAlternatingMap (f i)) (H …
    -/
  · rw [map_add, map_add, map_add, hx, hy]
    /-
      🎉 no goals
    -/
    /-
      case h.ι_mul
      R : Type u_1
      M : Type u_2
      N : Type u_3
      N' : Type u_4
      inst✝⁶ : CommRing R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : AddCommGroup N
      inst✝³ : AddCommGroup N'
      inst✝² : Module R M
      inst✝¹ : Module R N
      inst✝ : Module R N'
      g : LinearMap (RingHom.id R) N N'
      x : CliffordAlgebra 0
      m : M
      hx : ∀ (f : (i : Nat) → AlternatingMap R M N (Fin i)), Eq ((ExteriorAlgebra.li …
      f : (i : Nat) → AlternatingMap R M N (Fin i)
      ⊢ Eq ((ExteriorAlgebra.liftAlternating fun i => g.compAlternatingMap (f i)) (H …
    -/
  · rw [liftAlternating_ι_mul, liftAlternating_ι_mul, ← hx]
    /-
      case h.ι_mul
      R : Type u_1
      M : Type u_2
      N : Type u_3
      N' : Type u_4
      inst✝⁶ : CommRing R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : AddCommGroup N
      inst✝³ : AddCommGroup N'
      inst✝² : Module R M
      inst✝¹ : Module R N
      inst✝ : Module R N'
      g : LinearMap (RingHom.id R) N N'
      x : CliffordAlgebra 0
      m : M
      hx : ∀ (f : (i : Nat) → AlternatingMap R M N (Fin i)), Eq ((ExteriorAlgebra.li …
      f : (i : Nat) → AlternatingMap R M N (Fin i)
      ⊢ Eq ((ExteriorAlgebra.liftAlternating fun i => (g.compAlternatingMap (f i.suc …
    -/
    simp_rw [AlternatingMap.curryLeft_compAlternatingMap]
    /-
      🎉 no goals
    -/


@[simp]
theorem liftAlternating_ιMulti :
    liftAlternating (R := R) (M := M) (N := ExteriorAlgebra R M) (ιMulti R) =
    (LinearMap.id : ExteriorAlgebra R M →ₗ[R] ExteriorAlgebra R M) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ⊢ Eq (ExteriorAlgebra.liftAlternating (ExteriorAlgebra.ιMulti R)) LinearMap.id
  -/
  ext v
  /-
    case h
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    v : ExteriorAlgebra R M
    ⊢ Eq ((ExteriorAlgebra.liftAlternating (ExteriorAlgebra.ιMulti R)) v) (LinearM …
  -/
  dsimp
  /-
    case h
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    v : ExteriorAlgebra R M
    ⊢ Eq ((ExteriorAlgebra.liftAlternating (ExteriorAlgebra.ιMulti R)) v) v
  -/
  induction' v using CliffordAlgebra.left_induction with r x y hx hy x m hx
    /-
      case h.algebraMap
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      r : R
      ⊢ Eq ((ExteriorAlgebra.liftAlternating (ExteriorAlgebra.ιMulti R)) ((algebraMa …
    -/
  · rw [liftAlternating_algebraMap, ιMulti_zero_apply, Algebra.algebraMap_eq_smul_one]
    /-
      🎉 no goals
    -/
    /-
      case h.add
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      x y : CliffordAlgebra 0
      hx : Eq ((ExteriorAlgebra.liftAlternating (ExteriorAlgebra.ιMulti R)) x) x
      hy : Eq ((ExteriorAlgebra.liftAlternating (ExteriorAlgebra.ιMulti R)) y) y
      ⊢ Eq ((ExteriorAlgebra.liftAlternating (ExteriorAlgebra.ιMulti R)) (HAdd.hAdd  …
    -/
  · rw [map_add, hx, hy]
    /-
      🎉 no goals
    -/
  · simp_rw [liftAlternating_ι_mul, ιMulti_succ_curryLeft, liftAlternating_comp,
      LinearMap.comp_apply, LinearMap.mulLeft_apply, hx]


/-- `ExteriorAlgebra.liftAlternating` is an equivalence. -/
@[simps apply symm_apply]
def liftAlternatingEquiv : (∀ i, M [⋀^Fin i]→ₗ[R] N) ≃ₗ[R] ExteriorAlgebra R M →ₗ[R] N where
  toFun := liftAlternating (R := R)
  map_add' := map_add _
  map_smul' := map_smul _
  invFun F i := F.compAlternatingMap (ιMulti R i)
  left_inv _ := funext fun _ => liftAlternating_comp_ιMulti _
  right_inv F :=
                                           /-
                                             R : Type u_1
                                             M : Type u_2
                                             N : Type u_3
                                             N' : Type u_4
                                             inst✝⁶ : CommRing R
                                             inst✝⁵ : AddCommGroup M
                                             inst✝⁴ : AddCommGroup N
                                             inst✝³ : AddCommGroup N'
                                             inst✝² : Module R M
                                             inst✝¹ : Module R N
                                             inst✝ : Module R N'
                                             F : LinearMap (RingHom.id R) (ExteriorAlgebra R M) N
                                             ⊢ Eq (F.comp (ExteriorAlgebra.liftAlternating (ExteriorAlgebra.ιMulti R))) F
                                           -/
    (liftAlternating_comp _ _).trans <| by rw [liftAlternating_ιMulti, LinearMap.comp_id]
                                           /-
                                             🎉 no goals
                                           -/


/-- To show that two linear maps from the exterior algebra agree, it suffices to show they agree on
the exterior powers.

See note [partially-applied ext lemmas] -/
@[ext]
theorem lhom_ext ⦃f g : ExteriorAlgebra R M →ₗ[R] N⦄
    (h : ∀ i, f.compAlternatingMap (ιMulti R i) = g.compAlternatingMap (ιMulti R i)) : f = g :=
  liftAlternatingEquiv.symm.injective <| funext h


