theorem ContinuousLinearMap.norm_map_tail_le
    (f : Ei 0 →L[𝕜] ContinuousMultilinearMap 𝕜 (fun i : Fin n => Ei i.succ) G) (m : ∀ i, Ei i) :
    ‖f (m 0) (tail m)‖ ≤ ‖f‖ * ∏ i, ‖m i‖ :=
  calc
    ‖f (m 0) (tail m)‖ ≤ ‖f (m 0)‖ * ∏ i, ‖(tail m) i‖ := (f (m 0)).le_opNorm _
                                                                                          /-
                                                                                            𝕜 : Type u
                                                                                            n : Nat
                                                                                            Ei : Fin n.succ → Type wEi
                                                                                            G : Type wG
                                                                                            inst✝⁴ : NontriviallyNormedField 𝕜
                                                                                            inst✝³ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
                                                                                            inst✝² : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
                                                                                            inst✝¹ : NormedAddCommGroup G
                                                                                            inst✝ : NormedSpace 𝕜 G
                                                                                            f : ContinuousLinearMap (RingHom.id 𝕜) (Ei 0) (ContinuousMultilinearMap 𝕜 (fun …
                                                                                            m : (i : Fin n.succ) → Ei i
                                                                                            ⊢ LE.le 0 (Finset.univ.prod fun i => Norm.norm (Fin.tail m i))
                                                                                          -/
    _ ≤ ‖f‖ * ‖m 0‖ * ∏ i, ‖tail m i‖ := mul_le_mul_of_nonneg_right (f.le_opNorm _) <| by positivity
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/
                                                /-
                                                  𝕜 : Type u
                                                  n : Nat
                                                  Ei : Fin n.succ → Type wEi
                                                  G : Type wG
                                                  inst✝⁴ : NontriviallyNormedField 𝕜
                                                  inst✝³ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
                                                  inst✝² : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
                                                  inst✝¹ : NormedAddCommGroup G
                                                  inst✝ : NormedSpace 𝕜 G
                                                  f : ContinuousLinearMap (RingHom.id 𝕜) (Ei 0) (ContinuousMultilinearMap 𝕜 (fun …
                                                  m : (i : Fin n.succ) → Ei i
                                                  ⊢ Eq (HMul.hMul (HMul.hMul (Norm.norm f) (Norm.norm (m 0))) (Finset.univ.prod  …
                                                -/
    _ = ‖f‖ * (‖m 0‖ * ∏ i, ‖(tail m) i‖) := by ring
                                                /-
                                                  🎉 no goals
                                                -/
    _ = ‖f‖ * ∏ i, ‖m i‖ := by
      /-
        𝕜 : Type u
        n : Nat
        Ei : Fin n.succ → Type wEi
        G : Type wG
        inst✝⁴ : NontriviallyNormedField 𝕜
        inst✝³ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
        inst✝² : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        f : ContinuousLinearMap (RingHom.id 𝕜) (Ei 0) (ContinuousMultilinearMap 𝕜 (fun …
        m : (i : Fin n.succ) → Ei i
        ⊢ Eq (HMul.hMul (Norm.norm f) (HMul.hMul (Norm.norm (m 0)) (Finset.univ.prod f …
      -/
      rw [prod_univ_succ]
      /-
        𝕜 : Type u
        n : Nat
        Ei : Fin n.succ → Type wEi
        G : Type wG
        inst✝⁴ : NontriviallyNormedField 𝕜
        inst✝³ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
        inst✝² : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        f : ContinuousLinearMap (RingHom.id 𝕜) (Ei 0) (ContinuousMultilinearMap 𝕜 (fun …
        m : (i : Fin n.succ) → Ei i
        ⊢ Eq (HMul.hMul (Norm.norm f) (HMul.hMul (Norm.norm (m 0)) (Finset.univ.prod f …
      -/
      rfl
      /-
        🎉 no goals
      -/


theorem ContinuousMultilinearMap.norm_map_init_le
    (f : ContinuousMultilinearMap 𝕜 (fun i : Fin n => Ei <| castSucc i) (Ei (last n) →L[𝕜] G))
    (m : ∀ i, Ei i) : ‖f (init m) (m (last n))‖ ≤ ‖f‖ * ∏ i, ‖m i‖ :=
  calc
    ‖f (init m) (m (last n))‖ ≤ ‖f (init m)‖ * ‖m (last n)‖ := (f (init m)).le_opNorm _
    _ ≤ (‖f‖ * ∏ i, ‖(init m) i‖) * ‖m (last n)‖ :=
      (mul_le_mul_of_nonneg_right (f.le_opNorm _) (norm_nonneg _))
    _ = ‖f‖ * ((∏ i, ‖(init m) i‖) * ‖m (last n)‖) := mul_assoc _ _ _
    _ = ‖f‖ * ∏ i, ‖m i‖ := by
      /-
        𝕜 : Type u
        n : Nat
        Ei : Fin n.succ → Type wEi
        G : Type wG
        inst✝⁴ : NontriviallyNormedField 𝕜
        inst✝³ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
        inst✝² : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        f : ContinuousMultilinearMap 𝕜 (fun i => Ei i.castSucc) (ContinuousLinearMap ( …
        m : (i : Fin n.succ) → Ei i
        ⊢ Eq (HMul.hMul (Norm.norm f) (HMul.hMul (Finset.univ.prod fun i => Norm.norm  …
      -/
      rw [prod_univ_castSucc]
      /-
        𝕜 : Type u
        n : Nat
        Ei : Fin n.succ → Type wEi
        G : Type wG
        inst✝⁴ : NontriviallyNormedField 𝕜
        inst✝³ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
        inst✝² : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        f : ContinuousMultilinearMap 𝕜 (fun i => Ei i.castSucc) (ContinuousLinearMap ( …
        m : (i : Fin n.succ) → Ei i
        ⊢ Eq (HMul.hMul (Norm.norm f) (HMul.hMul (Finset.univ.prod fun i => Norm.norm  …
      -/
      rfl
      /-
        🎉 no goals
      -/


theorem ContinuousMultilinearMap.norm_map_cons_le (f : ContinuousMultilinearMap 𝕜 Ei G) (x : Ei 0)
    (m : ∀ i : Fin n, Ei i.succ) : ‖f (cons x m)‖ ≤ ‖f‖ * ‖x‖ * ∏ i, ‖m i‖ :=
  calc
    ‖f (cons x m)‖ ≤ ‖f‖ * ∏ i, ‖cons x m i‖ := f.le_opNorm _
    _ = ‖f‖ * ‖x‖ * ∏ i, ‖m i‖ := by
      /-
        𝕜 : Type u
        n : Nat
        Ei : Fin n.succ → Type wEi
        G : Type wG
        inst✝⁴ : NontriviallyNormedField 𝕜
        inst✝³ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
        inst✝² : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        f : ContinuousMultilinearMap 𝕜 Ei G
        x : Ei 0
        m : (i : Fin n) → Ei i.succ
        ⊢ Eq (HMul.hMul (Norm.norm f) (Finset.univ.prod fun i => Norm.norm (Fin.cons x …
      -/
      rw [prod_univ_succ]
      /-
        𝕜 : Type u
        n : Nat
        Ei : Fin n.succ → Type wEi
        G : Type wG
        inst✝⁴ : NontriviallyNormedField 𝕜
        inst✝³ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
        inst✝² : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        f : ContinuousMultilinearMap 𝕜 Ei G
        x : Ei 0
        m : (i : Fin n) → Ei i.succ
        ⊢ Eq (HMul.hMul (Norm.norm f) (HMul.hMul (Norm.norm (Fin.cons x m 0)) (Finset. …
      -/
      simp [mul_assoc]
      /-
        🎉 no goals
      -/


theorem ContinuousMultilinearMap.norm_map_snoc_le (f : ContinuousMultilinearMap 𝕜 Ei G)
    (m : ∀ i : Fin n, Ei <| castSucc i) (x : Ei (last n)) :
    ‖f (snoc m x)‖ ≤ (‖f‖ * ∏ i, ‖m i‖) * ‖x‖ :=
  calc
    ‖f (snoc m x)‖ ≤ ‖f‖ * ∏ i, ‖snoc m x i‖ := f.le_opNorm _
    _ = (‖f‖ * ∏ i, ‖m i‖) * ‖x‖ := by
      /-
        𝕜 : Type u
        n : Nat
        Ei : Fin n.succ → Type wEi
        G : Type wG
        inst✝⁴ : NontriviallyNormedField 𝕜
        inst✝³ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
        inst✝² : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        f : ContinuousMultilinearMap 𝕜 Ei G
        m : (i : Fin n) → Ei i.castSucc
        x : Ei (Fin.last n)
        ⊢ Eq (HMul.hMul (Norm.norm f) (Finset.univ.prod fun i => Norm.norm (Fin.snoc m …
      -/
      rw [prod_univ_castSucc]
      /-
        𝕜 : Type u
        n : Nat
        Ei : Fin n.succ → Type wEi
        G : Type wG
        inst✝⁴ : NontriviallyNormedField 𝕜
        inst✝³ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
        inst✝² : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        f : ContinuousMultilinearMap 𝕜 Ei G
        m : (i : Fin n) → Ei i.castSucc
        x : Ei (Fin.last n)
        ⊢ Eq (HMul.hMul (Norm.norm f) (HMul.hMul (Finset.univ.prod fun i => Norm.norm  …
      -/
      simp [mul_assoc]
      /-
        🎉 no goals
      -/


/-- Given a continuous linear map `f` from `E 0` to continuous multilinear maps on `n` variables,
construct the corresponding continuous multilinear map on `n+1` variables obtained by concatenating
the variables, given by `m ↦ f (m 0) (tail m)`-/
def ContinuousLinearMap.uncurryLeft
    (f : Ei 0 →L[𝕜] ContinuousMultilinearMap 𝕜 (fun i : Fin n => Ei i.succ) G) :
    ContinuousMultilinearMap 𝕜 Ei G :=
  (@LinearMap.uncurryLeft 𝕜 n Ei G _ _ _ _ _
      (ContinuousMultilinearMap.toMultilinearMapLinear.comp f.toLinearMap)).mkContinuous
                    /-
                      𝕜 : Type u
                      ι : Type v
                      ι' : Type v'
                      n : Nat
                      E : ι → Type wE
                      Ei : Fin n.succ → Type wEi
                      G : Type wG
                      G' : Type wG'
                      inst✝¹⁰ : Fintype ι
                      inst✝⁹ : Fintype ι'
                      inst✝⁸ : NontriviallyNormedField 𝕜
                      inst✝⁷ : (i : ι) → NormedAddCommGroup (E i)
                      inst✝⁶ : (i : ι) → NormedSpace 𝕜 (E i)
                      inst✝⁵ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
                      inst✝⁴ : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
                      inst✝³ : NormedAddCommGroup G
                      inst✝² : NormedSpace 𝕜 G
                      inst✝¹ : NormedAddCommGroup G'
                      inst✝ : NormedSpace 𝕜 G'
                      f : ContinuousLinearMap (RingHom.id 𝕜) (Ei 0) (ContinuousMultilinearMap 𝕜 (fun …
                      m : (i : Fin n.succ) → Ei i
                      ⊢ LE.le (Norm.norm ((ContinuousMultilinearMap.toMultilinearMapLinear.comp ↑f). …
                    -/
    ‖f‖ fun m => by exact ContinuousLinearMap.norm_map_tail_le f m
                    /-
                      🎉 no goals
                    -/


@[simp]
theorem ContinuousLinearMap.uncurryLeft_apply
    (f : Ei 0 →L[𝕜] ContinuousMultilinearMap 𝕜 (fun i : Fin n => Ei i.succ) G) (m : ∀ i, Ei i) :
    f.uncurryLeft m = f (m 0) (tail m) :=
  rfl


/-- Given a continuous multilinear map `f` in `n+1` variables, split the first variable to obtain
a continuous linear map into continuous multilinear maps in `n` variables, given by
`x ↦ (m ↦ f (cons x m))`. -/
def ContinuousMultilinearMap.curryLeft (f : ContinuousMultilinearMap 𝕜 Ei G) :
    Ei 0 →L[𝕜] ContinuousMultilinearMap 𝕜 (fun i : Fin n => Ei i.succ) G :=
  LinearMap.mkContinuous
    { -- define a linear map into `n` continuous multilinear maps
      -- from an `n+1` continuous multilinear map
      toFun := fun x =>
        (f.toMultilinearMap.curryLeft x).mkContinuous (‖f‖ * ‖x‖) (f.norm_map_cons_le x)
      map_add' := fun x y => by
        /-
          𝕜 : Type u
          ι : Type v
          ι' : Type v'
          n : Nat
          E : ι → Type wE
          Ei : Fin n.succ → Type wEi
          G : Type wG
          G' : Type wG'
          inst✝¹⁰ : Fintype ι
          inst✝⁹ : Fintype ι'
          inst✝⁸ : NontriviallyNormedField 𝕜
          inst✝⁷ : (i : ι) → NormedAddCommGroup (E i)
          inst✝⁶ : (i : ι) → NormedSpace 𝕜 (E i)
          inst✝⁵ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
          inst✝⁴ : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
          inst✝³ : NormedAddCommGroup G
          inst✝² : NormedSpace 𝕜 G
          inst✝¹ : NormedAddCommGroup G'
          inst✝ : NormedSpace 𝕜 G'
          f : ContinuousMultilinearMap 𝕜 Ei G
          x y : Ei 0
          ⊢ Eq ((fun x => (f.curryLeft x).mkContinuous (HMul.hMul (Norm.norm f) (Norm.no …
        -/
        ext m
        /-
          case H
          𝕜 : Type u
          ι : Type v
          ι' : Type v'
          n : Nat
          E : ι → Type wE
          Ei : Fin n.succ → Type wEi
          G : Type wG
          G' : Type wG'
          inst✝¹⁰ : Fintype ι
          inst✝⁹ : Fintype ι'
          inst✝⁸ : NontriviallyNormedField 𝕜
          inst✝⁷ : (i : ι) → NormedAddCommGroup (E i)
          inst✝⁶ : (i : ι) → NormedSpace 𝕜 (E i)
          inst✝⁵ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
          inst✝⁴ : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
          inst✝³ : NormedAddCommGroup G
          inst✝² : NormedSpace 𝕜 G
          inst✝¹ : NormedAddCommGroup G'
          inst✝ : NormedSpace 𝕜 G'
          f : ContinuousMultilinearMap 𝕜 Ei G
          x y : Ei 0
          m : (i : Fin n) → Ei i.succ
          ⊢ Eq (((fun x => (f.curryLeft x).mkContinuous (HMul.hMul (Norm.norm f) (Norm.n …
        -/
        exact f.cons_add m x y
        /-
          🎉 no goals
        -/
      map_smul' := fun c x => by
        /-
          𝕜 : Type u
          ι : Type v
          ι' : Type v'
          n : Nat
          E : ι → Type wE
          Ei : Fin n.succ → Type wEi
          G : Type wG
          G' : Type wG'
          inst✝¹⁰ : Fintype ι
          inst✝⁹ : Fintype ι'
          inst✝⁸ : NontriviallyNormedField 𝕜
          inst✝⁷ : (i : ι) → NormedAddCommGroup (E i)
          inst✝⁶ : (i : ι) → NormedSpace 𝕜 (E i)
          inst✝⁵ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
          inst✝⁴ : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
          inst✝³ : NormedAddCommGroup G
          inst✝² : NormedSpace 𝕜 G
          inst✝¹ : NormedAddCommGroup G'
          inst✝ : NormedSpace 𝕜 G'
          f : ContinuousMultilinearMap 𝕜 Ei G
          c : 𝕜
          x : Ei 0
          ⊢ Eq ({ toFun := fun x => (f.curryLeft x).mkContinuous (HMul.hMul (Norm.norm f …
        -/
        ext m
        exact
          f.cons_smul m c x }-- then register its continuity thanks to its boundedness properties.
    ‖f‖ fun x => by
      /-
        𝕜 : Type u
        ι : Type v
        ι' : Type v'
        n : Nat
        E : ι → Type wE
        Ei : Fin n.succ → Type wEi
        G : Type wG
        G' : Type wG'
        inst✝¹⁰ : Fintype ι
        inst✝⁹ : Fintype ι'
        inst✝⁸ : NontriviallyNormedField 𝕜
        inst✝⁷ : (i : ι) → NormedAddCommGroup (E i)
        inst✝⁶ : (i : ι) → NormedSpace 𝕜 (E i)
        inst✝⁵ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
        inst✝⁴ : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
        inst✝³ : NormedAddCommGroup G
        inst✝² : NormedSpace 𝕜 G
        inst✝¹ : NormedAddCommGroup G'
        inst✝ : NormedSpace 𝕜 G'
        f : ContinuousMultilinearMap 𝕜 Ei G
        x : Ei 0
        ⊢ LE.le (Norm.norm ({ toFun := fun x => (f.curryLeft x).mkContinuous (HMul.hMu …
      -/
      rw [LinearMap.coe_mk, AddHom.coe_mk]
      /-
        𝕜 : Type u
        ι : Type v
        ι' : Type v'
        n : Nat
        E : ι → Type wE
        Ei : Fin n.succ → Type wEi
        G : Type wG
        G' : Type wG'
        inst✝¹⁰ : Fintype ι
        inst✝⁹ : Fintype ι'
        inst✝⁸ : NontriviallyNormedField 𝕜
        inst✝⁷ : (i : ι) → NormedAddCommGroup (E i)
        inst✝⁶ : (i : ι) → NormedSpace 𝕜 (E i)
        inst✝⁵ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
        inst✝⁴ : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
        inst✝³ : NormedAddCommGroup G
        inst✝² : NormedSpace 𝕜 G
        inst✝¹ : NormedAddCommGroup G'
        inst✝ : NormedSpace 𝕜 G'
        f : ContinuousMultilinearMap 𝕜 Ei G
        x : Ei 0
        ⊢ LE.le (Norm.norm ((f.curryLeft x).mkContinuous (HMul.hMul (Norm.norm f) (Nor …
      -/
      exact MultilinearMap.mkContinuous_norm_le _ (mul_nonneg (norm_nonneg _) (norm_nonneg _)) _
      /-
        🎉 no goals
      -/


@[simp]
theorem ContinuousMultilinearMap.curryLeft_apply (f : ContinuousMultilinearMap 𝕜 Ei G) (x : Ei 0)
    (m : ∀ i : Fin n, Ei i.succ) : f.curryLeft x m = f (cons x m) :=
  rfl


@[simp]
theorem ContinuousLinearMap.curry_uncurryLeft
    (f : Ei 0 →L[𝕜] ContinuousMultilinearMap 𝕜 (fun i : Fin n => Ei i.succ) G) :
    f.uncurryLeft.curryLeft = f := by
  /-
    𝕜 : Type u
    n : Nat
    Ei : Fin n.succ → Type wEi
    G : Type wG
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
    inst✝² : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ContinuousLinearMap (RingHom.id 𝕜) (Ei 0) (ContinuousMultilinearMap 𝕜 (fun …
    ⊢ Eq f.uncurryLeft.curryLeft f
  -/
  ext m x
  rw [ContinuousMultilinearMap.curryLeft_apply, ContinuousLinearMap.uncurryLeft_apply, tail_cons,
    cons_zero]


@[simp]
theorem ContinuousMultilinearMap.uncurry_curryLeft (f : ContinuousMultilinearMap 𝕜 Ei G) :
    f.curryLeft.uncurryLeft = f :=
  ContinuousMultilinearMap.toMultilinearMap_injective <| f.toMultilinearMap.uncurry_curryLeft


/-- The space of continuous multilinear maps on `Π(i : Fin (n+1)), E i` is canonically isomorphic to
the space of continuous linear maps from `E 0` to the space of continuous multilinear maps on
`Π(i : Fin n), E i.succ`, by separating the first variable. We register this isomorphism in
`continuousMultilinearCurryLeftEquiv 𝕜 E E₂`. The algebraic version (without topology) is given
in `multilinearCurryLeftEquiv 𝕜 E E₂`.

The direct and inverse maps are given by `f.curryLeft` and `f.uncurryLeft`. Use these
unless you need the full framework of linear isometric equivs. -/
def continuousMultilinearCurryLeftEquiv :
    ContinuousMultilinearMap 𝕜 Ei G ≃ₗᵢ[𝕜]
      Ei 0 →L[𝕜] ContinuousMultilinearMap 𝕜 (fun i : Fin n => Ei i.succ) G :=
  LinearIsometryEquiv.ofBounds
    { toFun := ContinuousMultilinearMap.curryLeft
      map_add' := fun _ _ => rfl
      map_smul' := fun _ _ => rfl
      invFun := ContinuousLinearMap.uncurryLeft
      left_inv := ContinuousMultilinearMap.uncurry_curryLeft
      right_inv := ContinuousLinearMap.curry_uncurryLeft }
    (fun f => by
      /-
        𝕜 : Type u
        ι : Type v
        ι' : Type v'
        n : Nat
        E : ι → Type wE
        Ei : Fin n.succ → Type wEi
        G : Type wG
        G' : Type wG'
        inst✝¹⁰ : Fintype ι
        inst✝⁹ : Fintype ι'
        inst✝⁸ : NontriviallyNormedField 𝕜
        inst✝⁷ : (i : ι) → NormedAddCommGroup (E i)
        inst✝⁶ : (i : ι) → NormedSpace 𝕜 (E i)
        inst✝⁵ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
        inst✝⁴ : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
        inst✝³ : NormedAddCommGroup G
        inst✝² : NormedSpace 𝕜 G
        inst✝¹ : NormedAddCommGroup G'
        inst✝ : NormedSpace 𝕜 G'
        f : ContinuousMultilinearMap 𝕜 Ei G
        ⊢ LE.le (Norm.norm ({ toFun := ContinuousMultilinearMap.curryLeft, map_add' := …
      -/
      simp only [LinearEquiv.coe_mk]
      /-
        𝕜 : Type u
        ι : Type v
        ι' : Type v'
        n : Nat
        E : ι → Type wE
        Ei : Fin n.succ → Type wEi
        G : Type wG
        G' : Type wG'
        inst✝¹⁰ : Fintype ι
        inst✝⁹ : Fintype ι'
        inst✝⁸ : NontriviallyNormedField 𝕜
        inst✝⁷ : (i : ι) → NormedAddCommGroup (E i)
        inst✝⁶ : (i : ι) → NormedSpace 𝕜 (E i)
        inst✝⁵ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
        inst✝⁴ : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
        inst✝³ : NormedAddCommGroup G
        inst✝² : NormedSpace 𝕜 G
        inst✝¹ : NormedAddCommGroup G'
        inst✝ : NormedSpace 𝕜 G'
        f : ContinuousMultilinearMap 𝕜 Ei G
        ⊢ LE.le (Norm.norm f.curryLeft) (Norm.norm f)
      -/
      exact LinearMap.mkContinuous_norm_le _ (norm_nonneg f) _)
      /-
        🎉 no goals
      -/
    (fun f => by
      /-
        𝕜 : Type u
        ι : Type v
        ι' : Type v'
        n : Nat
        E : ι → Type wE
        Ei : Fin n.succ → Type wEi
        G : Type wG
        G' : Type wG'
        inst✝¹⁰ : Fintype ι
        inst✝⁹ : Fintype ι'
        inst✝⁸ : NontriviallyNormedField 𝕜
        inst✝⁷ : (i : ι) → NormedAddCommGroup (E i)
        inst✝⁶ : (i : ι) → NormedSpace 𝕜 (E i)
        inst✝⁵ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
        inst✝⁴ : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
        inst✝³ : NormedAddCommGroup G
        inst✝² : NormedSpace 𝕜 G
        inst✝¹ : NormedAddCommGroup G'
        inst✝ : NormedSpace 𝕜 G'
        f : ContinuousLinearMap (RingHom.id 𝕜) (Ei 0) (ContinuousMultilinearMap 𝕜 (fun …
        ⊢ LE.le (Norm.norm ({ toFun := ContinuousMultilinearMap.curryLeft, map_add' := …
      -/
      simp only [LinearEquiv.coe_symm_mk]
      /-
        𝕜 : Type u
        ι : Type v
        ι' : Type v'
        n : Nat
        E : ι → Type wE
        Ei : Fin n.succ → Type wEi
        G : Type wG
        G' : Type wG'
        inst✝¹⁰ : Fintype ι
        inst✝⁹ : Fintype ι'
        inst✝⁸ : NontriviallyNormedField 𝕜
        inst✝⁷ : (i : ι) → NormedAddCommGroup (E i)
        inst✝⁶ : (i : ι) → NormedSpace 𝕜 (E i)
        inst✝⁵ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
        inst✝⁴ : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
        inst✝³ : NormedAddCommGroup G
        inst✝² : NormedSpace 𝕜 G
        inst✝¹ : NormedAddCommGroup G'
        inst✝ : NormedSpace 𝕜 G'
        f : ContinuousLinearMap (RingHom.id 𝕜) (Ei 0) (ContinuousMultilinearMap 𝕜 (fun …
        ⊢ LE.le (Norm.norm f.uncurryLeft) (Norm.norm f)
      -/
      exact MultilinearMap.mkContinuous_norm_le _ (norm_nonneg f) _)
      /-
        🎉 no goals
      -/


@[simp]
theorem continuousMultilinearCurryLeftEquiv_apply
    (f : ContinuousMultilinearMap 𝕜 Ei G) (x : Ei 0) (v : Π i : Fin n, Ei i.succ) :
    continuousMultilinearCurryLeftEquiv 𝕜 Ei G f x v = f (cons x v) :=
  rfl


@[simp]
theorem continuousMultilinearCurryLeftEquiv_symm_apply
    (f : Ei 0 →L[𝕜] ContinuousMultilinearMap 𝕜 (fun i : Fin n => Ei i.succ) G) (v : Π i, Ei i) :
    (continuousMultilinearCurryLeftEquiv 𝕜 Ei G).symm f v = f (v 0) (tail v) :=
  rfl


@[simp]
theorem ContinuousMultilinearMap.curryLeft_norm (f : ContinuousMultilinearMap 𝕜 Ei G) :
    ‖f.curryLeft‖ = ‖f‖ :=
  (continuousMultilinearCurryLeftEquiv 𝕜 Ei G).norm_map f


@[simp]
theorem ContinuousLinearMap.uncurryLeft_norm
    (f : Ei 0 →L[𝕜] ContinuousMultilinearMap 𝕜 (fun i : Fin n => Ei i.succ) G) :
    ‖f.uncurryLeft‖ = ‖f‖ :=
  (continuousMultilinearCurryLeftEquiv 𝕜 Ei G).symm.norm_map f


/-- Given a continuous linear map `f` from continuous multilinear maps on `n` variables to
continuous linear maps on `E 0`, construct the corresponding continuous multilinear map on `n+1`
variables obtained by concatenating the variables, given by `m ↦ f (init m) (m (last n))`. -/
def ContinuousMultilinearMap.uncurryRight
    (f : ContinuousMultilinearMap 𝕜 (fun i : Fin n => Ei <| castSucc i) (Ei (last n) →L[𝕜] G)) :
    ContinuousMultilinearMap 𝕜 Ei G :=
  let f' : MultilinearMap 𝕜 (fun i : Fin n => Ei <| castSucc i) (Ei (last n) →ₗ[𝕜] G) :=
    { toFun := fun m => (f m).toLinearMap
                                           /-
                                             𝕜 : Type u
                                             ι : Type v
                                             ι' : Type v'
                                             n : Nat
                                             E : ι → Type wE
                                             Ei : Fin n.succ → Type wEi
                                             G : Type wG
                                             G' : Type wG'
                                             inst✝¹¹ : Fintype ι
                                             inst✝¹⁰ : Fintype ι'
                                             inst✝⁹ : NontriviallyNormedField 𝕜
                                             inst✝⁸ : (i : ι) → NormedAddCommGroup (E i)
                                             inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
                                             inst✝⁶ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
                                             inst✝⁵ : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
                                             inst✝⁴ : NormedAddCommGroup G
                                             inst✝³ : NormedSpace 𝕜 G
                                             inst✝² : NormedAddCommGroup G'
                                             inst✝¹ : NormedSpace 𝕜 G'
                                             f : ContinuousMultilinearMap 𝕜 (fun i => Ei i.castSucc) (ContinuousLinearMap ( …
                                             inst✝ : DecidableEq (Fin n)
                                             m : (i : Fin n) → Ei i.castSucc
                                             i : Fin n
                                             x y : Ei i.castSucc
                                             ⊢ Eq ((fun m => ↑(f m)) (Function.update m i (HAdd.hAdd x y))) (HAdd.hAdd ((fu …
                                           -/
      map_update_add' := fun m i x y => by simp
                                           /-
                                             🎉 no goals
                                           -/
                                            /-
                                              𝕜 : Type u
                                              ι : Type v
                                              ι' : Type v'
                                              n : Nat
                                              E : ι → Type wE
                                              Ei : Fin n.succ → Type wEi
                                              G : Type wG
                                              G' : Type wG'
                                              inst✝¹¹ : Fintype ι
                                              inst✝¹⁰ : Fintype ι'
                                              inst✝⁹ : NontriviallyNormedField 𝕜
                                              inst✝⁸ : (i : ι) → NormedAddCommGroup (E i)
                                              inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
                                              inst✝⁶ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
                                              inst✝⁵ : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
                                              inst✝⁴ : NormedAddCommGroup G
                                              inst✝³ : NormedSpace 𝕜 G
                                              inst✝² : NormedAddCommGroup G'
                                              inst✝¹ : NormedSpace 𝕜 G'
                                              f : ContinuousMultilinearMap 𝕜 (fun i => Ei i.castSucc) (ContinuousLinearMap ( …
                                              inst✝ : DecidableEq (Fin n)
                                              m : (i : Fin n) → Ei i.castSucc
                                              i : Fin n
                                              c : 𝕜
                                              x : Ei i.castSucc
                                              ⊢ Eq ((fun m => ↑(f m)) (Function.update m i (HSMul.hSMul c x))) (HSMul.hSMul  …
                                            -/
      map_update_smul' := fun m i c x => by simp }
                                            /-
                                              🎉 no goals
                                            -/
  (@MultilinearMap.uncurryRight 𝕜 n Ei G _ _ _ _ _ f').mkContinuous ‖f‖ fun m =>
    f.norm_map_init_le m


@[simp]
theorem ContinuousMultilinearMap.uncurryRight_apply
    (f : ContinuousMultilinearMap 𝕜 (fun i : Fin n => Ei <| castSucc i) (Ei (last n) →L[𝕜] G))
    (m : ∀ i, Ei i) : f.uncurryRight m = f (init m) (m (last n)) :=
  rfl


/-- Given a continuous multilinear map `f` in `n+1` variables, split the last variable to obtain
a continuous multilinear map in `n` variables into continuous linear maps, given by
`m ↦ (x ↦ f (snoc m x))`. -/
def ContinuousMultilinearMap.curryRight (f : ContinuousMultilinearMap 𝕜 Ei G) :
    ContinuousMultilinearMap 𝕜 (fun i : Fin n => Ei <| castSucc i) (Ei (last n) →L[𝕜] G) :=
  let f' : MultilinearMap 𝕜 (fun i : Fin n => Ei <| castSucc i) (Ei (last n) →L[𝕜] G) :=
    { toFun := fun m =>
        (f.toMultilinearMap.curryRight m).mkContinuous (‖f‖ * ∏ i, ‖m i‖) fun x =>
          f.norm_map_snoc_le m x
      map_update_add' := fun m i x y => by
        /-
          𝕜 : Type u
          ι : Type v
          ι' : Type v'
          n : Nat
          E : ι → Type wE
          Ei : Fin n.succ → Type wEi
          G : Type wG
          G' : Type wG'
          inst✝¹¹ : Fintype ι
          inst✝¹⁰ : Fintype ι'
          inst✝⁹ : NontriviallyNormedField 𝕜
          inst✝⁸ : (i : ι) → NormedAddCommGroup (E i)
          inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
          inst✝⁶ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
          inst✝⁵ : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
          inst✝⁴ : NormedAddCommGroup G
          inst✝³ : NormedSpace 𝕜 G
          inst✝² : NormedAddCommGroup G'
          inst✝¹ : NormedSpace 𝕜 G'
          f : ContinuousMultilinearMap 𝕜 Ei G
          inst✝ : DecidableEq (Fin n)
          m : (i : Fin n) → Ei i.castSucc
          i : Fin n
          x y : Ei i.castSucc
          ⊢ Eq ((fun m => (f.curryRight m).mkContinuous (HMul.hMul (Norm.norm f) (Finset …
        -/
        ext
        /-
          case h
          𝕜 : Type u
          ι : Type v
          ι' : Type v'
          n : Nat
          E : ι → Type wE
          Ei : Fin n.succ → Type wEi
          G : Type wG
          G' : Type wG'
          inst✝¹¹ : Fintype ι
          inst✝¹⁰ : Fintype ι'
          inst✝⁹ : NontriviallyNormedField 𝕜
          inst✝⁸ : (i : ι) → NormedAddCommGroup (E i)
          inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
          inst✝⁶ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
          inst✝⁵ : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
          inst✝⁴ : NormedAddCommGroup G
          inst✝³ : NormedSpace 𝕜 G
          inst✝² : NormedAddCommGroup G'
          inst✝¹ : NormedSpace 𝕜 G'
          f : ContinuousMultilinearMap 𝕜 Ei G
          inst✝ : DecidableEq (Fin n)
          m : (i : Fin n) → Ei i.castSucc
          i : Fin n
          x y : Ei i.castSucc
          x✝ : Ei (Fin.last n)
          ⊢ Eq (((fun m => (f.curryRight m).mkContinuous (HMul.hMul (Norm.norm f) (Finse …
        -/
        simp
        /-
          🎉 no goals
        -/
      map_update_smul' := fun m i c x => by
        /-
          𝕜 : Type u
          ι : Type v
          ι' : Type v'
          n : Nat
          E : ι → Type wE
          Ei : Fin n.succ → Type wEi
          G : Type wG
          G' : Type wG'
          inst✝¹¹ : Fintype ι
          inst✝¹⁰ : Fintype ι'
          inst✝⁹ : NontriviallyNormedField 𝕜
          inst✝⁸ : (i : ι) → NormedAddCommGroup (E i)
          inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
          inst✝⁶ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
          inst✝⁵ : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
          inst✝⁴ : NormedAddCommGroup G
          inst✝³ : NormedSpace 𝕜 G
          inst✝² : NormedAddCommGroup G'
          inst✝¹ : NormedSpace 𝕜 G'
          f : ContinuousMultilinearMap 𝕜 Ei G
          inst✝ : DecidableEq (Fin n)
          m : (i : Fin n) → Ei i.castSucc
          i : Fin n
          c : 𝕜
          x : Ei i.castSucc
          ⊢ Eq ((fun m => (f.curryRight m).mkContinuous (HMul.hMul (Norm.norm f) (Finset …
        -/
        ext
        /-
          case h
          𝕜 : Type u
          ι : Type v
          ι' : Type v'
          n : Nat
          E : ι → Type wE
          Ei : Fin n.succ → Type wEi
          G : Type wG
          G' : Type wG'
          inst✝¹¹ : Fintype ι
          inst✝¹⁰ : Fintype ι'
          inst✝⁹ : NontriviallyNormedField 𝕜
          inst✝⁸ : (i : ι) → NormedAddCommGroup (E i)
          inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
          inst✝⁶ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
          inst✝⁵ : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
          inst✝⁴ : NormedAddCommGroup G
          inst✝³ : NormedSpace 𝕜 G
          inst✝² : NormedAddCommGroup G'
          inst✝¹ : NormedSpace 𝕜 G'
          f : ContinuousMultilinearMap 𝕜 Ei G
          inst✝ : DecidableEq (Fin n)
          m : (i : Fin n) → Ei i.castSucc
          i : Fin n
          c : 𝕜
          x : Ei i.castSucc
          x✝ : Ei (Fin.last n)
          ⊢ Eq (((fun m => (f.curryRight m).mkContinuous (HMul.hMul (Norm.norm f) (Finse …
        -/
        simp }
        /-
          🎉 no goals
        -/
  f'.mkContinuous ‖f‖ fun m => by
    /-
      𝕜 : Type u
      ι : Type v
      ι' : Type v'
      n : Nat
      E : ι → Type wE
      Ei : Fin n.succ → Type wEi
      G : Type wG
      G' : Type wG'
      inst✝¹⁰ : Fintype ι
      inst✝⁹ : Fintype ι'
      inst✝⁸ : NontriviallyNormedField 𝕜
      inst✝⁷ : (i : ι) → NormedAddCommGroup (E i)
      inst✝⁶ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝⁵ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
      inst✝⁴ : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
      inst✝³ : NormedAddCommGroup G
      inst✝² : NormedSpace 𝕜 G
      inst✝¹ : NormedAddCommGroup G'
      inst✝ : NormedSpace 𝕜 G'
      f : ContinuousMultilinearMap 𝕜 Ei G
      f' : MultilinearMap 𝕜 (fun i => Ei i.castSucc) (ContinuousLinearMap (RingHom.i …
      m : (i : Fin n) → Ei i.castSucc
      ⊢ LE.le (Norm.norm (f' m)) (HMul.hMul (Norm.norm f) (Finset.univ.prod fun i => …
    -/
    simp only [f', MultilinearMap.coe_mk]
    /-
      𝕜 : Type u
      ι : Type v
      ι' : Type v'
      n : Nat
      E : ι → Type wE
      Ei : Fin n.succ → Type wEi
      G : Type wG
      G' : Type wG'
      inst✝¹⁰ : Fintype ι
      inst✝⁹ : Fintype ι'
      inst✝⁸ : NontriviallyNormedField 𝕜
      inst✝⁷ : (i : ι) → NormedAddCommGroup (E i)
      inst✝⁶ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝⁵ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
      inst✝⁴ : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
      inst✝³ : NormedAddCommGroup G
      inst✝² : NormedSpace 𝕜 G
      inst✝¹ : NormedAddCommGroup G'
      inst✝ : NormedSpace 𝕜 G'
      f : ContinuousMultilinearMap 𝕜 Ei G
      f' : MultilinearMap 𝕜 (fun i => Ei i.castSucc) (ContinuousLinearMap (RingHom.i …
      m : (i : Fin n) → Ei i.castSucc
      ⊢ LE.le (Norm.norm ((f.curryRight m).mkContinuous (HMul.hMul (Norm.norm f) (Fi …
    -/
    exact LinearMap.mkContinuous_norm_le _ (by positivity) _
    /-
      🎉 no goals
    -/


@[simp]
theorem ContinuousMultilinearMap.curryRight_apply (f : ContinuousMultilinearMap 𝕜 Ei G)
    (m : ∀ i : Fin n, Ei <| castSucc i) (x : Ei (last n)) : f.curryRight m x = f (snoc m x) :=
  rfl


@[simp]
theorem ContinuousMultilinearMap.curry_uncurryRight
    (f : ContinuousMultilinearMap 𝕜 (fun i : Fin n => Ei <| castSucc i) (Ei (last n) →L[𝕜] G)) :
    f.uncurryRight.curryRight = f := by
  /-
    𝕜 : Type u
    n : Nat
    Ei : Fin n.succ → Type wEi
    G : Type wG
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
    inst✝² : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ContinuousMultilinearMap 𝕜 (fun i => Ei i.castSucc) (ContinuousLinearMap ( …
    ⊢ Eq f.uncurryRight.curryRight f
  -/
  ext m x
  rw [ContinuousMultilinearMap.curryRight_apply, ContinuousMultilinearMap.uncurryRight_apply,
    snoc_last, init_snoc]


@[simp]
theorem ContinuousMultilinearMap.uncurry_curryRight (f : ContinuousMultilinearMap 𝕜 Ei G) :
    f.curryRight.uncurryRight = f := by
  /-
    𝕜 : Type u
    n : Nat
    Ei : Fin n.succ → Type wEi
    G : Type wG
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
    inst✝² : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ContinuousMultilinearMap 𝕜 Ei G
    ⊢ Eq f.curryRight.uncurryRight f
  -/
  ext m
  /-
    case H
    𝕜 : Type u
    n : Nat
    Ei : Fin n.succ → Type wEi
    G : Type wG
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
    inst✝² : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ContinuousMultilinearMap 𝕜 Ei G
    m : (i : Fin n.succ) → Ei i
    ⊢ Eq (f.curryRight.uncurryRight m) (f m)
  -/
  rw [uncurryRight_apply, curryRight_apply, snoc_init_self]
  /-
    🎉 no goals
  -/


/--
The space of continuous multilinear maps on `Π(i : Fin (n+1)), Ei i` is canonically isomorphic to
the space of continuous multilinear maps on `Π(i : Fin n), Ei <| castSucc i` with values in the
space of continuous linear maps on `Ei (last n)`, by separating the last variable. We register this
isomorphism as a continuous linear equiv in `continuousMultilinearCurryRightEquiv 𝕜 Ei G`.
The algebraic version (without topology) is given in `multilinearCurryRightEquiv 𝕜 Ei G`.

The direct and inverse maps are given by `f.curryRight` and `f.uncurryRight`. Use these
unless you need the full framework of linear isometric equivs.
-/
def continuousMultilinearCurryRightEquiv :
    ContinuousMultilinearMap 𝕜 Ei G ≃ₗᵢ[𝕜]
      ContinuousMultilinearMap 𝕜 (fun i : Fin n => Ei <| castSucc i) (Ei (last n) →L[𝕜] G) :=
  LinearIsometryEquiv.ofBounds
    { toFun := ContinuousMultilinearMap.curryRight
      map_add' := fun _ _ => rfl
      map_smul' := fun _ _ => rfl
      invFun := ContinuousMultilinearMap.uncurryRight
      left_inv := ContinuousMultilinearMap.uncurry_curryRight
      right_inv := ContinuousMultilinearMap.curry_uncurryRight }
    (fun f => by
      /-
        𝕜 : Type u
        ι : Type v
        ι' : Type v'
        n : Nat
        E : ι → Type wE
        Ei : Fin n.succ → Type wEi
        G : Type wG
        G' : Type wG'
        inst✝¹⁰ : Fintype ι
        inst✝⁹ : Fintype ι'
        inst✝⁸ : NontriviallyNormedField 𝕜
        inst✝⁷ : (i : ι) → NormedAddCommGroup (E i)
        inst✝⁶ : (i : ι) → NormedSpace 𝕜 (E i)
        inst✝⁵ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
        inst✝⁴ : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
        inst✝³ : NormedAddCommGroup G
        inst✝² : NormedSpace 𝕜 G
        inst✝¹ : NormedAddCommGroup G'
        inst✝ : NormedSpace 𝕜 G'
        f : ContinuousMultilinearMap 𝕜 Ei G
        ⊢ LE.le (Norm.norm ({ toFun := ContinuousMultilinearMap.curryRight, map_add' : …
      -/
      simp only [curryRight, LinearEquiv.coe_mk]
      /-
        𝕜 : Type u
        ι : Type v
        ι' : Type v'
        n : Nat
        E : ι → Type wE
        Ei : Fin n.succ → Type wEi
        G : Type wG
        G' : Type wG'
        inst✝¹⁰ : Fintype ι
        inst✝⁹ : Fintype ι'
        inst✝⁸ : NontriviallyNormedField 𝕜
        inst✝⁷ : (i : ι) → NormedAddCommGroup (E i)
        inst✝⁶ : (i : ι) → NormedSpace 𝕜 (E i)
        inst✝⁵ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
        inst✝⁴ : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
        inst✝³ : NormedAddCommGroup G
        inst✝² : NormedSpace 𝕜 G
        inst✝¹ : NormedAddCommGroup G'
        inst✝ : NormedSpace 𝕜 G'
        f : ContinuousMultilinearMap 𝕜 Ei G
        ⊢ LE.le (Norm.norm ({ toFun := fun m => (f.curryRight m).mkContinuous (HMul.hM …
      -/
      exact MultilinearMap.mkContinuous_norm_le _ (norm_nonneg f) _)
      /-
        🎉 no goals
      -/
    (fun f => by
      /-
        𝕜 : Type u
        ι : Type v
        ι' : Type v'
        n : Nat
        E : ι → Type wE
        Ei : Fin n.succ → Type wEi
        G : Type wG
        G' : Type wG'
        inst✝¹⁰ : Fintype ι
        inst✝⁹ : Fintype ι'
        inst✝⁸ : NontriviallyNormedField 𝕜
        inst✝⁷ : (i : ι) → NormedAddCommGroup (E i)
        inst✝⁶ : (i : ι) → NormedSpace 𝕜 (E i)
        inst✝⁵ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
        inst✝⁴ : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
        inst✝³ : NormedAddCommGroup G
        inst✝² : NormedSpace 𝕜 G
        inst✝¹ : NormedAddCommGroup G'
        inst✝ : NormedSpace 𝕜 G'
        f : ContinuousMultilinearMap 𝕜 (fun i => Ei i.castSucc) (ContinuousLinearMap ( …
        ⊢ LE.le (Norm.norm ({ toFun := ContinuousMultilinearMap.curryRight, map_add' : …
      -/
      simp only [uncurryRight, LinearEquiv.coe_symm_mk]
      /-
        𝕜 : Type u
        ι : Type v
        ι' : Type v'
        n : Nat
        E : ι → Type wE
        Ei : Fin n.succ → Type wEi
        G : Type wG
        G' : Type wG'
        inst✝¹⁰ : Fintype ι
        inst✝⁹ : Fintype ι'
        inst✝⁸ : NontriviallyNormedField 𝕜
        inst✝⁷ : (i : ι) → NormedAddCommGroup (E i)
        inst✝⁶ : (i : ι) → NormedSpace 𝕜 (E i)
        inst✝⁵ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
        inst✝⁴ : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
        inst✝³ : NormedAddCommGroup G
        inst✝² : NormedSpace 𝕜 G
        inst✝¹ : NormedAddCommGroup G'
        inst✝ : NormedSpace 𝕜 G'
        f : ContinuousMultilinearMap 𝕜 (fun i => Ei i.castSucc) (ContinuousLinearMap ( …
        ⊢ LE.le (Norm.norm ({ toFun := fun m => ↑(f m), map_update_add' := ⋯, map_upda …
      -/
      exact MultilinearMap.mkContinuous_norm_le _ (norm_nonneg f) _)
      /-
        🎉 no goals
      -/


/-- The space of continuous multilinear maps on `Π(i : Fin (n+1)), G` is canonically isomorphic to
the space of continuous multilinear maps on `Π(i : Fin n), G` with values in the space
of continuous linear maps on `G`, by separating the last variable. We register this
isomorphism as a continuous linear equiv in `continuousMultilinearCurryRightEquiv' 𝕜 n G G'`.
For a version allowing dependent types, see `continuousMultilinearCurryRightEquiv`. When there
are no dependent types, use the primed version as it helps Lean a lot for unification.

The direct and inverse maps are given by `f.curryRight` and `f.uncurryRight`. Use these
unless you need the full framework of linear isometric equivs. -/
def continuousMultilinearCurryRightEquiv' : (G[×n.succ]→L[𝕜] G') ≃ₗᵢ[𝕜] G[×n]→L[𝕜] G →L[𝕜] G' :=
  continuousMultilinearCurryRightEquiv 𝕜 (fun _ => G) G'


@[simp]
theorem continuousMultilinearCurryRightEquiv_apply
    (f : ContinuousMultilinearMap 𝕜 Ei G) (v : Π i : Fin n, Ei <| castSucc i) (x : Ei (last n)) :
    continuousMultilinearCurryRightEquiv 𝕜 Ei G f v x = f (snoc v x) :=
  rfl


@[simp]
theorem continuousMultilinearCurryRightEquiv_symm_apply
    (f : ContinuousMultilinearMap 𝕜 (fun i : Fin n => Ei <| castSucc i) (Ei (last n) →L[𝕜] G))
    (v : Π i, Ei i) :
    (continuousMultilinearCurryRightEquiv 𝕜 Ei G).symm f v = f (init v) (v (last n)) :=
  rfl


@[simp]
theorem continuousMultilinearCurryRightEquiv_apply'
    (f : G[×n.succ]→L[𝕜] G') (v : Fin n → G) (x : G) :
    continuousMultilinearCurryRightEquiv' 𝕜 n G G' f v x = f (snoc v x) :=
  rfl


@[simp]
theorem continuousMultilinearCurryRightEquiv_symm_apply'
    (f : G[×n]→L[𝕜] G →L[𝕜] G') (v : Fin (n + 1) → G) :
    (continuousMultilinearCurryRightEquiv' 𝕜 n G G').symm f v = f (init v) (v (last n)) :=
  rfl


@[simp]
theorem ContinuousMultilinearMap.curryRight_norm (f : ContinuousMultilinearMap 𝕜 Ei G) :
    ‖f.curryRight‖ = ‖f‖ :=
  (continuousMultilinearCurryRightEquiv 𝕜 Ei G).norm_map f


@[simp]
theorem ContinuousMultilinearMap.uncurryRight_norm
    (f : ContinuousMultilinearMap 𝕜 (fun i : Fin n => Ei <| castSucc i) (Ei (last n) →L[𝕜] G)) :
    ‖f.uncurryRight‖ = ‖f‖ :=
  (continuousMultilinearCurryRightEquiv 𝕜 Ei G).symm.norm_map f


/-- Associating to a continuous multilinear map in `0` variables the unique value it takes. -/
def ContinuousMultilinearMap.curry0 (f : ContinuousMultilinearMap 𝕜 (fun _ : Fin 0 => G) G') :
    G' :=
  f 0


variable (𝕜 G) in
/-- Associating to an element `x` of a vector space `E₂` the continuous multilinear map in `0`
variables taking the (unique) value `x` -/
def ContinuousMultilinearMap.uncurry0 (x : G') : G[×0]→L[𝕜] G' :=
  ContinuousMultilinearMap.constOfIsEmpty 𝕜 _ x


variable (𝕜) in
@[simp]
theorem ContinuousMultilinearMap.uncurry0_apply (x : G') (m : Fin 0 → G) :
    ContinuousMultilinearMap.uncurry0 𝕜 G x m = x :=
  rfl


@[simp]
theorem ContinuousMultilinearMap.curry0_apply (f : G[×0]→L[𝕜] G') : f.curry0 = f 0 :=
  rfl


@[simp]
theorem ContinuousMultilinearMap.apply_zero_uncurry0 (f : G[×0]→L[𝕜] G') {x : Fin 0 → G} :
    ContinuousMultilinearMap.uncurry0 𝕜 G (f x) = f := by
  /-
    𝕜 : Type u
    G : Type wG
    G' : Type wG'
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup G'
    inst✝ : NormedSpace 𝕜 G'
    f : ContinuousMultilinearMap 𝕜 (fun i => G) G'
    x : Fin 0 → G
    ⊢ Eq (ContinuousMultilinearMap.uncurry0 𝕜 G (f x)) f
  -/
  ext m
  /-
    case H
    𝕜 : Type u
    G : Type wG
    G' : Type wG'
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup G'
    inst✝ : NormedSpace 𝕜 G'
    f : ContinuousMultilinearMap 𝕜 (fun i => G) G'
    x m : Fin 0 → G
    ⊢ Eq ((ContinuousMultilinearMap.uncurry0 𝕜 G (f x)) m) (f m)
  -/
  simp [Subsingleton.elim x m]
  /-
    🎉 no goals
  -/


theorem ContinuousMultilinearMap.uncurry0_curry0 (f : G[×0]→L[𝕜] G') :
                                                             /-
                                                               𝕜 : Type u
                                                               G : Type wG
                                                               G' : Type wG'
                                                               inst✝⁴ : NontriviallyNormedField 𝕜
                                                               inst✝³ : NormedAddCommGroup G
                                                               inst✝² : NormedSpace 𝕜 G
                                                               inst✝¹ : NormedAddCommGroup G'
                                                               inst✝ : NormedSpace 𝕜 G'
                                                               f : ContinuousMultilinearMap 𝕜 (fun i => G) G'
                                                               ⊢ Eq (ContinuousMultilinearMap.uncurry0 𝕜 G f.curry0) f
                                                             -/
    ContinuousMultilinearMap.uncurry0 𝕜 G f.curry0 = f := by simp
                                                             /-
                                                               🎉 no goals
                                                             -/


variable (𝕜 G) in
theorem ContinuousMultilinearMap.curry0_uncurry0 (x : G') :
    (ContinuousMultilinearMap.uncurry0 𝕜 G x).curry0 = x :=
  rfl


variable (𝕜 G) in
@[simp]
theorem ContinuousMultilinearMap.uncurry0_norm (x : G') :
    ‖ContinuousMultilinearMap.uncurry0 𝕜 G x‖ = ‖x‖ :=
  norm_constOfIsEmpty _ _ _


@[simp]
theorem ContinuousMultilinearMap.fin0_apply_norm (f : G[×0]→L[𝕜] G') {x : Fin 0 → G} :
    ‖f x‖ = ‖f‖ := by
  /-
    𝕜 : Type u
    G : Type wG
    G' : Type wG'
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup G'
    inst✝ : NormedSpace 𝕜 G'
    f : ContinuousMultilinearMap 𝕜 (fun i => G) G'
    x : Fin 0 → G
    ⊢ Eq (Norm.norm (f x)) (Norm.norm f)
  -/
  obtain rfl : x = 0 := Subsingleton.elim _ _
  /-
    𝕜 : Type u
    G : Type wG
    G' : Type wG'
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup G'
    inst✝ : NormedSpace 𝕜 G'
    f : ContinuousMultilinearMap 𝕜 (fun i => G) G'
    ⊢ Eq (Norm.norm (f 0)) (Norm.norm f)
  -/
  refine le_antisymm (by simpa using f.le_opNorm 0) ?_
  have : ‖ContinuousMultilinearMap.uncurry0 𝕜 G f.curry0‖ ≤ ‖f.curry0‖ :=
    ContinuousMultilinearMap.opNorm_le_bound (norm_nonneg _) fun m => by
      simp [-ContinuousMultilinearMap.apply_zero_uncurry0]
  /-
    𝕜 : Type u
    G : Type wG
    G' : Type wG'
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup G'
    inst✝ : NormedSpace 𝕜 G'
    f : ContinuousMultilinearMap 𝕜 (fun i => G) G'
    this : LE.le (Norm.norm (ContinuousMultilinearMap.uncurry0 𝕜 G f.curry0)) (Nor …
    ⊢ LE.le (Norm.norm f) (Norm.norm (f 0))
  -/
  simpa [-Matrix.zero_empty] using this
  /-
    🎉 no goals
  -/


                                                                                          /-
                                                                                            𝕜 : Type u
                                                                                            G : Type wG
                                                                                            G' : Type wG'
                                                                                            inst✝⁴ : NontriviallyNormedField 𝕜
                                                                                            inst✝³ : NormedAddCommGroup G
                                                                                            inst✝² : NormedSpace 𝕜 G
                                                                                            inst✝¹ : NormedAddCommGroup G'
                                                                                            inst✝ : NormedSpace 𝕜 G'
                                                                                            f : ContinuousMultilinearMap 𝕜 (fun i => G) G'
                                                                                            ⊢ Eq (Norm.norm f.curry0) (Norm.norm f)
                                                                                          -/
theorem ContinuousMultilinearMap.curry0_norm (f : G[×0]→L[𝕜] G') : ‖f.curry0‖ = ‖f‖ := by simp
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


/-- The continuous linear isomorphism between elements of a normed space, and continuous multilinear
maps in `0` variables with values in this normed space.

The direct and inverse maps are `uncurry0` and `curry0`. Use these unless you need the full
framework of linear isometric equivs. -/
def continuousMultilinearCurryFin0 : (G[×0]→L[𝕜] G') ≃ₗᵢ[𝕜] G' where
  toFun f := ContinuousMultilinearMap.curry0 f
  invFun f := ContinuousMultilinearMap.uncurry0 𝕜 G f
  map_add' _ _ := rfl
  map_smul' _ _ := rfl
  left_inv := ContinuousMultilinearMap.uncurry0_curry0
  right_inv := ContinuousMultilinearMap.curry0_uncurry0 𝕜 G
  norm_map' := ContinuousMultilinearMap.curry0_norm


@[simp]
theorem continuousMultilinearCurryFin0_apply (f : G[×0]→L[𝕜] G') :
    continuousMultilinearCurryFin0 𝕜 G G' f = f 0 :=
  rfl


@[simp]
theorem continuousMultilinearCurryFin0_symm_apply (x : G') (v : Fin 0 → G) :
    (continuousMultilinearCurryFin0 𝕜 G G').symm x v = x :=
  rfl


/-- Continuous multilinear maps from `G^1` to `G'` are isomorphic with continuous linear maps from
`G` to `G'`. -/
def continuousMultilinearCurryFin1 : (G[×1]→L[𝕜] G') ≃ₗᵢ[𝕜] G →L[𝕜] G' :=
  (continuousMultilinearCurryRightEquiv 𝕜 (fun _ : Fin 1 => G) G').trans
    (continuousMultilinearCurryFin0 𝕜 G (G →L[𝕜] G'))


@[simp]
theorem continuousMultilinearCurryFin1_apply (f : G[×1]→L[𝕜] G') (x : G) :
    continuousMultilinearCurryFin1 𝕜 G G' f x = f (Fin.snoc 0 x) :=
  rfl


@[simp]
theorem continuousMultilinearCurryFin1_symm_apply (f : G →L[𝕜] G') (v : Fin 1 → G) :
    (continuousMultilinearCurryFin1 𝕜 G G').symm f v = f (v 0) :=
  rfl


@[simp]
theorem norm_domDomCongr (σ : ι ≃ ι') (f : ContinuousMultilinearMap 𝕜 (fun _ : ι => G) G') :
    ‖domDomCongr σ f‖ = ‖f‖ := by
  simp only [norm_def, LinearEquiv.coe_mk, ← σ.prod_comp,
    (σ.arrowCongr (Equiv.refl G)).surjective.forall, domDomCongr_apply, Equiv.arrowCongr_apply,
    Equiv.coe_refl, id_comp, comp_apply, Equiv.symm_apply_apply, id]


/-- An equivalence of the index set defines a linear isometric equivalence between the spaces
of multilinear maps. -/
def domDomCongrₗᵢ (σ : ι ≃ ι') :
    ContinuousMultilinearMap 𝕜 (fun _ : ι => G) G' ≃ₗᵢ[𝕜]
      ContinuousMultilinearMap 𝕜 (fun _ : ι' => G) G' :=
  { domDomCongrEquiv σ with
    map_add' := fun _ _ => rfl
    map_smul' := fun _ _ => rfl
    norm_map' := norm_domDomCongr 𝕜 G G' σ }


/-- A continuous multilinear map with variables indexed by `ι ⊕ ι'` defines a continuous
multilinear map with variables indexed by `ι` taking values in the space of continuous multilinear
maps with variables indexed by `ι'`. -/
def currySum (f : ContinuousMultilinearMap 𝕜 (fun _ : ι ⊕ ι' => G) G') :
    ContinuousMultilinearMap 𝕜 (fun _ : ι => G) (ContinuousMultilinearMap 𝕜 (fun _ : ι' => G) G') :=
  MultilinearMap.mkContinuousMultilinear (MultilinearMap.currySum f.toMultilinearMap) ‖f‖
                   /-
                     𝕜 : Type u
                     ι : Type v
                     ι' : Type v'
                     n : Nat
                     E : ι → Type wE
                     Ei : Fin n.succ → Type wEi
                     G : Type wG
                     G' : Type wG'
                     inst✝¹⁰ : Fintype ι
                     inst✝⁹ : Fintype ι'
                     inst✝⁸ : NontriviallyNormedField 𝕜
                     inst✝⁷ : (i : ι) → NormedAddCommGroup (E i)
                     inst✝⁶ : (i : ι) → NormedSpace 𝕜 (E i)
                     inst✝⁵ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
                     inst✝⁴ : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
                     inst✝³ : NormedAddCommGroup G
                     inst✝² : NormedSpace 𝕜 G
                     inst✝¹ : NormedAddCommGroup G'
                     inst✝ : NormedSpace 𝕜 G'
                     f : ContinuousMultilinearMap 𝕜 (fun x => G) G'
                     m : ι → G
                     m' : ι' → G
                     ⊢ LE.le (Norm.norm ((f.currySum m) m')) (HMul.hMul (HMul.hMul (Norm.norm f) (F …
                   -/
    fun m m' => by simpa [Fintype.prod_sum_type, mul_assoc] using f.le_opNorm (Sum.elim m m')
                   /-
                     🎉 no goals
                   -/


@[simp]
theorem currySum_apply (f : ContinuousMultilinearMap 𝕜 (fun _ : ι ⊕ ι' => G) G') (m : ι → G)
    (m' : ι' → G) : f.currySum m m' = f (Sum.elim m m') :=
  rfl


/-- A continuous multilinear map with variables indexed by `ι` taking values in the space of
continuous multilinear maps with variables indexed by `ι'` defines a continuous multilinear map with
variables indexed by `ι ⊕ ι'`. -/
def uncurrySum (f : ContinuousMultilinearMap 𝕜 (fun _ : ι => G)
    (ContinuousMultilinearMap 𝕜 (fun _ : ι' => G) G')) :
    ContinuousMultilinearMap 𝕜 (fun _ : ι ⊕ ι' => G) G' :=
  MultilinearMap.mkContinuous
    (toMultilinearMapLinear.compMultilinearMap f.toMultilinearMap).uncurrySum ‖f‖ fun m => by
    simpa [Fintype.prod_sum_type, mul_assoc] using
      (f (m ∘ Sum.inl)).le_of_opNorm_le (f.le_opNorm _) (m ∘ Sum.inr)


@[simp]
theorem uncurrySum_apply (f : ContinuousMultilinearMap 𝕜 (fun _ : ι => G)
    (ContinuousMultilinearMap 𝕜 (fun _ : ι' => G) G'))
    (m : ι ⊕ ι' → G) : f.uncurrySum m = f (m ∘ Sum.inl) (m ∘ Sum.inr) :=
  rfl


/-- Linear isometric equivalence between the space of continuous multilinear maps with variables
indexed by `ι ⊕ ι'` and the space of continuous multilinear maps with variables indexed by `ι`
taking values in the space of continuous multilinear maps with variables indexed by `ι'`.

The forward and inverse functions are `ContinuousMultilinearMap.currySum`
and `ContinuousMultilinearMap.uncurrySum`. Use this definition only if you need
some properties of `LinearIsometryEquiv`. -/
def currySumEquiv : ContinuousMultilinearMap 𝕜 (fun _ : ι ⊕ ι' => G) G' ≃ₗᵢ[𝕜]
    ContinuousMultilinearMap 𝕜 (fun _ : ι => G) (ContinuousMultilinearMap 𝕜 (fun _ : ι' => G) G') :=
  LinearIsometryEquiv.ofBounds
    { toFun := currySum
      invFun := uncurrySum
      map_add' := fun f g => by
        /-
          𝕜 : Type u
          ι : Type v
          ι' : Type v'
          n : Nat
          E : ι → Type wE
          Ei : Fin n.succ → Type wEi
          G : Type wG
          G' : Type wG'
          inst✝¹⁰ : Fintype ι
          inst✝⁹ : Fintype ι'
          inst✝⁸ : NontriviallyNormedField 𝕜
          inst✝⁷ : (i : ι) → NormedAddCommGroup (E i)
          inst✝⁶ : (i : ι) → NormedSpace 𝕜 (E i)
          inst✝⁵ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
          inst✝⁴ : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
          inst✝³ : NormedAddCommGroup G
          inst✝² : NormedSpace 𝕜 G
          inst✝¹ : NormedAddCommGroup G'
          inst✝ : NormedSpace 𝕜 G'
          f g : ContinuousMultilinearMap 𝕜 (fun x => G) G'
          ⊢ Eq (HAdd.hAdd f g).currySum (HAdd.hAdd f.currySum g.currySum)
        -/
        ext
        /-
          case H.H
          𝕜 : Type u
          ι : Type v
          ι' : Type v'
          n : Nat
          E : ι → Type wE
          Ei : Fin n.succ → Type wEi
          G : Type wG
          G' : Type wG'
          inst✝¹⁰ : Fintype ι
          inst✝⁹ : Fintype ι'
          inst✝⁸ : NontriviallyNormedField 𝕜
          inst✝⁷ : (i : ι) → NormedAddCommGroup (E i)
          inst✝⁶ : (i : ι) → NormedSpace 𝕜 (E i)
          inst✝⁵ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
          inst✝⁴ : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
          inst✝³ : NormedAddCommGroup G
          inst✝² : NormedSpace 𝕜 G
          inst✝¹ : NormedAddCommGroup G'
          inst✝ : NormedSpace 𝕜 G'
          f g : ContinuousMultilinearMap 𝕜 (fun x => G) G'
          x✝¹ : ι → G
          x✝ : ι' → G
          ⊢ Eq (((HAdd.hAdd f g).currySum x✝¹) x✝) (((HAdd.hAdd f.currySum g.currySum) x …
        -/
        rfl
        /-
          🎉 no goals
        -/
      map_smul' := fun c f => by
        /-
          𝕜 : Type u
          ι : Type v
          ι' : Type v'
          n : Nat
          E : ι → Type wE
          Ei : Fin n.succ → Type wEi
          G : Type wG
          G' : Type wG'
          inst✝¹⁰ : Fintype ι
          inst✝⁹ : Fintype ι'
          inst✝⁸ : NontriviallyNormedField 𝕜
          inst✝⁷ : (i : ι) → NormedAddCommGroup (E i)
          inst✝⁶ : (i : ι) → NormedSpace 𝕜 (E i)
          inst✝⁵ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
          inst✝⁴ : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
          inst✝³ : NormedAddCommGroup G
          inst✝² : NormedSpace 𝕜 G
          inst✝¹ : NormedAddCommGroup G'
          inst✝ : NormedSpace 𝕜 G'
          c : 𝕜
          f : ContinuousMultilinearMap 𝕜 (fun x => G) G'
          ⊢ Eq ({ toFun := ContinuousMultilinearMap.currySum, map_add' := ⋯ }.toFun (HSM …
        -/
        ext
        /-
          case H.H
          𝕜 : Type u
          ι : Type v
          ι' : Type v'
          n : Nat
          E : ι → Type wE
          Ei : Fin n.succ → Type wEi
          G : Type wG
          G' : Type wG'
          inst✝¹⁰ : Fintype ι
          inst✝⁹ : Fintype ι'
          inst✝⁸ : NontriviallyNormedField 𝕜
          inst✝⁷ : (i : ι) → NormedAddCommGroup (E i)
          inst✝⁶ : (i : ι) → NormedSpace 𝕜 (E i)
          inst✝⁵ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
          inst✝⁴ : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
          inst✝³ : NormedAddCommGroup G
          inst✝² : NormedSpace 𝕜 G
          inst✝¹ : NormedAddCommGroup G'
          inst✝ : NormedSpace 𝕜 G'
          c : 𝕜
          f : ContinuousMultilinearMap 𝕜 (fun x => G) G'
          x✝¹ : ι → G
          x✝ : ι' → G
          ⊢ Eq ((({ toFun := ContinuousMultilinearMap.currySum, map_add' := ⋯ }.toFun (H …
        -/
        rfl
        /-
          🎉 no goals
        -/
      left_inv := fun f => by
        /-
          𝕜 : Type u
          ι : Type v
          ι' : Type v'
          n : Nat
          E : ι → Type wE
          Ei : Fin n.succ → Type wEi
          G : Type wG
          G' : Type wG'
          inst✝¹⁰ : Fintype ι
          inst✝⁹ : Fintype ι'
          inst✝⁸ : NontriviallyNormedField 𝕜
          inst✝⁷ : (i : ι) → NormedAddCommGroup (E i)
          inst✝⁶ : (i : ι) → NormedSpace 𝕜 (E i)
          inst✝⁵ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
          inst✝⁴ : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
          inst✝³ : NormedAddCommGroup G
          inst✝² : NormedSpace 𝕜 G
          inst✝¹ : NormedAddCommGroup G'
          inst✝ : NormedSpace 𝕜 G'
          f : ContinuousMultilinearMap 𝕜 (fun x => G) G'
          ⊢ Eq ({ toFun := ContinuousMultilinearMap.currySum, map_add' := ⋯, map_smul' : …
        -/
        ext m
        /-
          case H
          𝕜 : Type u
          ι : Type v
          ι' : Type v'
          n : Nat
          E : ι → Type wE
          Ei : Fin n.succ → Type wEi
          G : Type wG
          G' : Type wG'
          inst✝¹⁰ : Fintype ι
          inst✝⁹ : Fintype ι'
          inst✝⁸ : NontriviallyNormedField 𝕜
          inst✝⁷ : (i : ι) → NormedAddCommGroup (E i)
          inst✝⁶ : (i : ι) → NormedSpace 𝕜 (E i)
          inst✝⁵ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
          inst✝⁴ : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
          inst✝³ : NormedAddCommGroup G
          inst✝² : NormedSpace 𝕜 G
          inst✝¹ : NormedAddCommGroup G'
          inst✝ : NormedSpace 𝕜 G'
          f : ContinuousMultilinearMap 𝕜 (fun x => G) G'
          m : Sum ι ι' → G
          ⊢ Eq (({ toFun := ContinuousMultilinearMap.currySum, map_add' := ⋯, map_smul'  …
        -/
        exact congr_arg f (Sum.elim_comp_inl_inr m)
        /-
          🎉 no goals
        -/
      right_inv := fun f => by
        /-
          𝕜 : Type u
          ι : Type v
          ι' : Type v'
          n : Nat
          E : ι → Type wE
          Ei : Fin n.succ → Type wEi
          G : Type wG
          G' : Type wG'
          inst✝¹⁰ : Fintype ι
          inst✝⁹ : Fintype ι'
          inst✝⁸ : NontriviallyNormedField 𝕜
          inst✝⁷ : (i : ι) → NormedAddCommGroup (E i)
          inst✝⁶ : (i : ι) → NormedSpace 𝕜 (E i)
          inst✝⁵ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
          inst✝⁴ : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
          inst✝³ : NormedAddCommGroup G
          inst✝² : NormedSpace 𝕜 G
          inst✝¹ : NormedAddCommGroup G'
          inst✝ : NormedSpace 𝕜 G'
          f : ContinuousMultilinearMap 𝕜 (fun x => G) (ContinuousMultilinearMap 𝕜 (fun x …
          ⊢ Eq ({ toFun := ContinuousMultilinearMap.currySum, map_add' := ⋯, map_smul' : …
        -/
        ext m₁ m₂
        /-
          case H.H
          𝕜 : Type u
          ι : Type v
          ι' : Type v'
          n : Nat
          E : ι → Type wE
          Ei : Fin n.succ → Type wEi
          G : Type wG
          G' : Type wG'
          inst✝¹⁰ : Fintype ι
          inst✝⁹ : Fintype ι'
          inst✝⁸ : NontriviallyNormedField 𝕜
          inst✝⁷ : (i : ι) → NormedAddCommGroup (E i)
          inst✝⁶ : (i : ι) → NormedSpace 𝕜 (E i)
          inst✝⁵ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
          inst✝⁴ : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
          inst✝³ : NormedAddCommGroup G
          inst✝² : NormedSpace 𝕜 G
          inst✝¹ : NormedAddCommGroup G'
          inst✝ : NormedSpace 𝕜 G'
          f : ContinuousMultilinearMap 𝕜 (fun x => G) (ContinuousMultilinearMap 𝕜 (fun x …
          m₁ : ι → G
          m₂ : ι' → G
          ⊢ Eq ((({ toFun := ContinuousMultilinearMap.currySum, map_add' := ⋯, map_smul' …
        -/
        rfl }
        /-
          🎉 no goals
        -/
    (fun f => MultilinearMap.mkContinuousMultilinear_norm_le _ (norm_nonneg f) _) fun f => by
      /-
        𝕜 : Type u
        ι : Type v
        ι' : Type v'
        n : Nat
        E : ι → Type wE
        Ei : Fin n.succ → Type wEi
        G : Type wG
        G' : Type wG'
        inst✝¹⁰ : Fintype ι
        inst✝⁹ : Fintype ι'
        inst✝⁸ : NontriviallyNormedField 𝕜
        inst✝⁷ : (i : ι) → NormedAddCommGroup (E i)
        inst✝⁶ : (i : ι) → NormedSpace 𝕜 (E i)
        inst✝⁵ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
        inst✝⁴ : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
        inst✝³ : NormedAddCommGroup G
        inst✝² : NormedSpace 𝕜 G
        inst✝¹ : NormedAddCommGroup G'
        inst✝ : NormedSpace 𝕜 G'
        f : ContinuousMultilinearMap 𝕜 (fun x => G) (ContinuousMultilinearMap 𝕜 (fun x …
        ⊢ LE.le (Norm.norm ({ toFun := ContinuousMultilinearMap.currySum, map_add' :=  …
      -/
      simp only [LinearEquiv.coe_symm_mk]
      /-
        𝕜 : Type u
        ι : Type v
        ι' : Type v'
        n : Nat
        E : ι → Type wE
        Ei : Fin n.succ → Type wEi
        G : Type wG
        G' : Type wG'
        inst✝¹⁰ : Fintype ι
        inst✝⁹ : Fintype ι'
        inst✝⁸ : NontriviallyNormedField 𝕜
        inst✝⁷ : (i : ι) → NormedAddCommGroup (E i)
        inst✝⁶ : (i : ι) → NormedSpace 𝕜 (E i)
        inst✝⁵ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
        inst✝⁴ : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
        inst✝³ : NormedAddCommGroup G
        inst✝² : NormedSpace 𝕜 G
        inst✝¹ : NormedAddCommGroup G'
        inst✝ : NormedSpace 𝕜 G'
        f : ContinuousMultilinearMap 𝕜 (fun x => G) (ContinuousMultilinearMap 𝕜 (fun x …
        ⊢ LE.le (Norm.norm f.uncurrySum) (Norm.norm f)
      -/
      exact MultilinearMap.mkContinuous_norm_le _ (norm_nonneg f) _
      /-
        🎉 no goals
      -/


/-- If `s : Finset (Fin n)` is a finite set of cardinality `k` and its complement has cardinality
`l`, then the space of continuous multilinear maps `G [×n]→L[𝕜] G'` of `n` variables is isomorphic
to the space of continuous multilinear maps `G [×k]→L[𝕜] G [×l]→L[𝕜] G'` of `k` variables taking
values in the space of continuous multilinear maps of `l` variables. -/
def curryFinFinset {k l n : ℕ} {s : Finset (Fin n)} (hk : #s = k) (hl : #sᶜ = l) :
    (G[×n]→L[𝕜] G') ≃ₗᵢ[𝕜] G[×k]→L[𝕜] G[×l]→L[𝕜] G' :=
  (domDomCongrₗᵢ 𝕜 G G' (finSumEquivOfFinset hk hl).symm).trans
    (currySumEquiv 𝕜 (Fin k) (Fin l) G G')


@[simp]
theorem curryFinFinset_apply (hk : #s = k) (hl : #sᶜ = l) (f : G[×n]→L[𝕜] G')
    (mk : Fin k → G) (ml : Fin l → G) : curryFinFinset 𝕜 G G' hk hl f mk ml =
      f fun i => Sum.elim mk ml ((finSumEquivOfFinset hk hl).symm i) :=
  rfl


@[simp]
theorem curryFinFinset_symm_apply (hk : #s = k) (hl : #sᶜ = l)
    (f : G[×k]→L[𝕜] G[×l]→L[𝕜] G') (m : Fin n → G) : (curryFinFinset 𝕜 G G' hk hl).symm f m =
      f (fun i => m <| finSumEquivOfFinset hk hl (Sum.inl i)) fun i =>
        m <| finSumEquivOfFinset hk hl (Sum.inr i) :=
  rfl


theorem curryFinFinset_symm_apply_piecewise_const (hk : #s = k) (hl : #sᶜ = l)
    (f : G[×k]→L[𝕜] G[×l]→L[𝕜] G') (x y : G) :
    (curryFinFinset 𝕜 G G' hk hl).symm f (s.piecewise (fun _ => x) fun _ => y) =
      f (fun _ => x) fun _ => y :=
  MultilinearMap.curryFinFinset_symm_apply_piecewise_const hk hl _ x y


@[simp]
theorem curryFinFinset_symm_apply_const (hk : #s = k) (hl : #sᶜ = l)
    (f : G[×k]→L[𝕜] G[×l]→L[𝕜] G') (x : G) :
    ((curryFinFinset 𝕜 G G' hk hl).symm f fun _ => x) = f (fun _ => x) fun _ => x :=
  rfl


theorem curryFinFinset_apply_const (hk : #s = k) (hl : #sᶜ = l) (f : G[×n]→L[𝕜] G')
    (x y : G) : (curryFinFinset 𝕜 G G' hk hl f (fun _ => x) fun _ => y) =
      f (s.piecewise (fun _ => x) fun _ => y) := by
  /-
    𝕜 : Type u
    n : Nat
    G : Type wG
    G' : Type wG'
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup G'
    inst✝ : NormedSpace 𝕜 G'
    k l : Nat
    s : Finset (Fin n)
    hk : Eq s.card k
    hl : Eq (HasCompl.compl s).card l
    f : ContinuousMultilinearMap 𝕜 (fun i => G) G'
    x y : G
    ⊢ Eq ((((ContinuousMultilinearMap.curryFinFinset 𝕜 G G' hk hl) f) fun x_1 => x …
  -/
  refine (curryFinFinset_symm_apply_piecewise_const hk hl _ _ _).symm.trans ?_
  /-
    𝕜 : Type u
    n : Nat
    G : Type wG
    G' : Type wG'
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup G'
    inst✝ : NormedSpace 𝕜 G'
    k l : Nat
    s : Finset (Fin n)
    hk : Eq s.card k
    hl : Eq (HasCompl.compl s).card l
    f : ContinuousMultilinearMap 𝕜 (fun i => G) G'
    x y : G
    ⊢ Eq (((ContinuousMultilinearMap.curryFinFinset 𝕜 G G' hk hl).symm ((Continuou …
  -/
  rw [LinearIsometryEquiv.symm_apply_apply]
  /-
    🎉 no goals
  -/


/-- Given a linear map into continuous multilinear maps
`B : G →L[𝕜] ContinuousMultilinearMap 𝕜 E F`, one can not always uncurry it as `G` and `E` might
live in a different universe. However, one can always lift it to a continuous multilinear map
on `(G × (Π i, E i)) ^ (1 + n)`, which maps `(v_0, ..., v_n)` to `B (g_0) (u_1, ..., u_n)` where
`g_0` is the `G`-coordinate of `v_0` and `u_i` is the `E_i` coordinate of `v_i`. -/
noncomputable def continuousMultilinearMapOption (B : G →L[𝕜] ContinuousMultilinearMap 𝕜 E F) :
    ContinuousMultilinearMap 𝕜 (fun (_ : Option ι) ↦ (G × (Π i, E i))) F :=
  MultilinearMap.mkContinuous
  { toFun := fun p ↦ B (p none).1 (fun i ↦ (p i).2 i)
    map_update_add' := by
      /-
        𝕜 : Type u
        ι : Type v
        ι' : Type v'
        n : Nat
        E : ι → Type wE
        Ei : Fin n.succ → Type wEi
        G✝ : Type wG
        G' : Type wG'
        inst✝¹⁴ : Fintype ι
        inst✝¹³ : Fintype ι'
        inst✝¹² : NontriviallyNormedField 𝕜
        inst✝¹¹ : (i : ι) → NormedAddCommGroup (E i)
        inst✝¹⁰ : (i : ι) → NormedSpace 𝕜 (E i)
        inst✝⁹ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
        inst✝⁸ : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
        inst✝⁷ : NormedAddCommGroup G✝
        inst✝⁶ : NormedSpace 𝕜 G✝
        inst✝⁵ : NormedAddCommGroup G'
        inst✝⁴ : NormedSpace 𝕜 G'
        F : Type u_1
        G : Type u_2
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        B : ContinuousLinearMap (RingHom.id 𝕜) G (ContinuousMultilinearMap 𝕜 E F)
        ⊢ ∀ [inst : DecidableEq (Option ι)] (m : Option ι → Prod G ((i : ι) → E i)) (i …
      -/
      intro inst v j x y
      match j with
      | none => simp
      | some j =>
        classical
        have B z : (fun i ↦ (Function.update v (some j) z (some i)).2 i) =
            Function.update (fun (i : ι) ↦ (v i).2 i) j (z.2 j) := by
          ext i
          rcases eq_or_ne i j with rfl | hij
          · simp
          · simp [hij]
        simp [B]
    map_update_smul' := by
      /-
        𝕜 : Type u
        ι : Type v
        ι' : Type v'
        n : Nat
        E : ι → Type wE
        Ei : Fin n.succ → Type wEi
        G✝ : Type wG
        G' : Type wG'
        inst✝¹⁴ : Fintype ι
        inst✝¹³ : Fintype ι'
        inst✝¹² : NontriviallyNormedField 𝕜
        inst✝¹¹ : (i : ι) → NormedAddCommGroup (E i)
        inst✝¹⁰ : (i : ι) → NormedSpace 𝕜 (E i)
        inst✝⁹ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
        inst✝⁸ : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
        inst✝⁷ : NormedAddCommGroup G✝
        inst✝⁶ : NormedSpace 𝕜 G✝
        inst✝⁵ : NormedAddCommGroup G'
        inst✝⁴ : NormedSpace 𝕜 G'
        F : Type u_1
        G : Type u_2
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        B : ContinuousLinearMap (RingHom.id 𝕜) G (ContinuousMultilinearMap 𝕜 E F)
        ⊢ ∀ [inst : DecidableEq (Option ι)] (m : Option ι → Prod G ((i : ι) → E i)) (i …
      -/
      intro inst v j c x
      match j with
      | none => simp
      | some j =>
        classical
        have B z : (fun i ↦ (Function.update v (some j) z (some i)).2 i) =
            Function.update (fun (i : ι) ↦ (v i).2 i) j (z.2 j) := by
          ext i
          rcases eq_or_ne i j with rfl | hij
          · simp
          · simp [hij]
        simp [B] } (‖B‖) <| by
  /-
    𝕜 : Type u
    ι : Type v
    ι' : Type v'
    n : Nat
    E : ι → Type wE
    Ei : Fin n.succ → Type wEi
    G✝ : Type wG
    G' : Type wG'
    inst✝¹⁴ : Fintype ι
    inst✝¹³ : Fintype ι'
    inst✝¹² : NontriviallyNormedField 𝕜
    inst✝¹¹ : (i : ι) → NormedAddCommGroup (E i)
    inst✝¹⁰ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝⁹ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
    inst✝⁸ : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
    inst✝⁷ : NormedAddCommGroup G✝
    inst✝⁶ : NormedSpace 𝕜 G✝
    inst✝⁵ : NormedAddCommGroup G'
    inst✝⁴ : NormedSpace 𝕜 G'
    F : Type u_1
    G : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) G (ContinuousMultilinearMap 𝕜 E F)
    ⊢ ∀ (m : Option ι → Prod G ((i : ι) → E i)), LE.le (Norm.norm ({ toFun := fun  …
  -/
  intro b
  /-
    𝕜 : Type u
    ι : Type v
    ι' : Type v'
    n : Nat
    E : ι → Type wE
    Ei : Fin n.succ → Type wEi
    G✝ : Type wG
    G' : Type wG'
    inst✝¹⁴ : Fintype ι
    inst✝¹³ : Fintype ι'
    inst✝¹² : NontriviallyNormedField 𝕜
    inst✝¹¹ : (i : ι) → NormedAddCommGroup (E i)
    inst✝¹⁰ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝⁹ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
    inst✝⁸ : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
    inst✝⁷ : NormedAddCommGroup G✝
    inst✝⁶ : NormedSpace 𝕜 G✝
    inst✝⁵ : NormedAddCommGroup G'
    inst✝⁴ : NormedSpace 𝕜 G'
    F : Type u_1
    G : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) G (ContinuousMultilinearMap 𝕜 E F)
    b : Option ι → Prod G ((i : ι) → E i)
    ⊢ LE.le (Norm.norm ({ toFun := fun p => (B (p Option.none).1) fun i => (p (Opt …
  -/
  simp only [MultilinearMap.coe_mk, Fintype.prod_option]
  /-
    𝕜 : Type u
    ι : Type v
    ι' : Type v'
    n : Nat
    E : ι → Type wE
    Ei : Fin n.succ → Type wEi
    G✝ : Type wG
    G' : Type wG'
    inst✝¹⁴ : Fintype ι
    inst✝¹³ : Fintype ι'
    inst✝¹² : NontriviallyNormedField 𝕜
    inst✝¹¹ : (i : ι) → NormedAddCommGroup (E i)
    inst✝¹⁰ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝⁹ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
    inst✝⁸ : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
    inst✝⁷ : NormedAddCommGroup G✝
    inst✝⁶ : NormedSpace 𝕜 G✝
    inst✝⁵ : NormedAddCommGroup G'
    inst✝⁴ : NormedSpace 𝕜 G'
    F : Type u_1
    G : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) G (ContinuousMultilinearMap 𝕜 E F)
    b : Option ι → Prod G ((i : ι) → E i)
    ⊢ LE.le (Norm.norm ((B (b Option.none).1) fun i => (b (Option.some i)).2 i)) ( …
  -/
  apply (ContinuousMultilinearMap.le_opNorm _ _).trans
  /-
    𝕜 : Type u
    ι : Type v
    ι' : Type v'
    n : Nat
    E : ι → Type wE
    Ei : Fin n.succ → Type wEi
    G✝ : Type wG
    G' : Type wG'
    inst✝¹⁴ : Fintype ι
    inst✝¹³ : Fintype ι'
    inst✝¹² : NontriviallyNormedField 𝕜
    inst✝¹¹ : (i : ι) → NormedAddCommGroup (E i)
    inst✝¹⁰ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝⁹ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
    inst✝⁸ : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
    inst✝⁷ : NormedAddCommGroup G✝
    inst✝⁶ : NormedSpace 𝕜 G✝
    inst✝⁵ : NormedAddCommGroup G'
    inst✝⁴ : NormedSpace 𝕜 G'
    F : Type u_1
    G : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) G (ContinuousMultilinearMap 𝕜 E F)
    b : Option ι → Prod G ((i : ι) → E i)
    ⊢ LE.le (HMul.hMul (Norm.norm (B (b Option.none).1)) (Finset.univ.prod fun i = …
  -/
  rw [← mul_assoc]
  /-
    𝕜 : Type u
    ι : Type v
    ι' : Type v'
    n : Nat
    E : ι → Type wE
    Ei : Fin n.succ → Type wEi
    G✝ : Type wG
    G' : Type wG'
    inst✝¹⁴ : Fintype ι
    inst✝¹³ : Fintype ι'
    inst✝¹² : NontriviallyNormedField 𝕜
    inst✝¹¹ : (i : ι) → NormedAddCommGroup (E i)
    inst✝¹⁰ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝⁹ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
    inst✝⁸ : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
    inst✝⁷ : NormedAddCommGroup G✝
    inst✝⁶ : NormedSpace 𝕜 G✝
    inst✝⁵ : NormedAddCommGroup G'
    inst✝⁴ : NormedSpace 𝕜 G'
    F : Type u_1
    G : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) G (ContinuousMultilinearMap 𝕜 E F)
    b : Option ι → Prod G ((i : ι) → E i)
    ⊢ LE.le (HMul.hMul (Norm.norm (B (b Option.none).1)) (Finset.univ.prod fun i = …
  -/
  gcongr with i _
    /-
      case h₁
      𝕜 : Type u
      ι : Type v
      ι' : Type v'
      n : Nat
      E : ι → Type wE
      Ei : Fin n.succ → Type wEi
      G✝ : Type wG
      G' : Type wG'
      inst✝¹⁴ : Fintype ι
      inst✝¹³ : Fintype ι'
      inst✝¹² : NontriviallyNormedField 𝕜
      inst✝¹¹ : (i : ι) → NormedAddCommGroup (E i)
      inst✝¹⁰ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝⁹ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
      inst✝⁸ : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
      inst✝⁷ : NormedAddCommGroup G✝
      inst✝⁶ : NormedSpace 𝕜 G✝
      inst✝⁵ : NormedAddCommGroup G'
      inst✝⁴ : NormedSpace 𝕜 G'
      F : Type u_1
      G : Type u_2
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      B : ContinuousLinearMap (RingHom.id 𝕜) G (ContinuousMultilinearMap 𝕜 E F)
      b : Option ι → Prod G ((i : ι) → E i)
      ⊢ LE.le (Norm.norm (B (b Option.none).1)) (HMul.hMul (Norm.norm B) (Norm.norm  …
    -/
  · apply (B.le_opNorm _).trans
    /-
      case h₁
      𝕜 : Type u
      ι : Type v
      ι' : Type v'
      n : Nat
      E : ι → Type wE
      Ei : Fin n.succ → Type wEi
      G✝ : Type wG
      G' : Type wG'
      inst✝¹⁴ : Fintype ι
      inst✝¹³ : Fintype ι'
      inst✝¹² : NontriviallyNormedField 𝕜
      inst✝¹¹ : (i : ι) → NormedAddCommGroup (E i)
      inst✝¹⁰ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝⁹ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
      inst✝⁸ : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
      inst✝⁷ : NormedAddCommGroup G✝
      inst✝⁶ : NormedSpace 𝕜 G✝
      inst✝⁵ : NormedAddCommGroup G'
      inst✝⁴ : NormedSpace 𝕜 G'
      F : Type u_1
      G : Type u_2
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      B : ContinuousLinearMap (RingHom.id 𝕜) G (ContinuousMultilinearMap 𝕜 E F)
      b : Option ι → Prod G ((i : ι) → E i)
      ⊢ LE.le (HMul.hMul (Norm.norm B) (Norm.norm (b Option.none).1)) (HMul.hMul (No …
    -/
    gcongr
    /-
      case h₁.h
      𝕜 : Type u
      ι : Type v
      ι' : Type v'
      n : Nat
      E : ι → Type wE
      Ei : Fin n.succ → Type wEi
      G✝ : Type wG
      G' : Type wG'
      inst✝¹⁴ : Fintype ι
      inst✝¹³ : Fintype ι'
      inst✝¹² : NontriviallyNormedField 𝕜
      inst✝¹¹ : (i : ι) → NormedAddCommGroup (E i)
      inst✝¹⁰ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝⁹ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
      inst✝⁸ : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
      inst✝⁷ : NormedAddCommGroup G✝
      inst✝⁶ : NormedSpace 𝕜 G✝
      inst✝⁵ : NormedAddCommGroup G'
      inst✝⁴ : NormedSpace 𝕜 G'
      F : Type u_1
      G : Type u_2
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      B : ContinuousLinearMap (RingHom.id 𝕜) G (ContinuousMultilinearMap 𝕜 E F)
      b : Option ι → Prod G ((i : ι) → E i)
      ⊢ LE.le (Norm.norm (b Option.none).1) (Norm.norm (b Option.none))
    -/
    exact norm_fst_le _
    /-
      🎉 no goals
    -/
    /-
      case h₂.h1
      𝕜 : Type u
      ι : Type v
      ι' : Type v'
      n : Nat
      E : ι → Type wE
      Ei : Fin n.succ → Type wEi
      G✝ : Type wG
      G' : Type wG'
      inst✝¹⁴ : Fintype ι
      inst✝¹³ : Fintype ι'
      inst✝¹² : NontriviallyNormedField 𝕜
      inst✝¹¹ : (i : ι) → NormedAddCommGroup (E i)
      inst✝¹⁰ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝⁹ : (i : Fin n.succ) → NormedAddCommGroup (Ei i)
      inst✝⁸ : (i : Fin n.succ) → NormedSpace 𝕜 (Ei i)
      inst✝⁷ : NormedAddCommGroup G✝
      inst✝⁶ : NormedSpace 𝕜 G✝
      inst✝⁵ : NormedAddCommGroup G'
      inst✝⁴ : NormedSpace 𝕜 G'
      F : Type u_1
      G : Type u_2
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      B : ContinuousLinearMap (RingHom.id 𝕜) G (ContinuousMultilinearMap 𝕜 E F)
      b : Option ι → Prod G ((i : ι) → E i)
      i : ι
      a✝ : Membership.mem Finset.univ i
      ⊢ LE.le (Norm.norm ((b (Option.some i)).2 i)) (Norm.norm (b (Option.some i)))
    -/
  · exact (norm_le_pi_norm _ _).trans (norm_snd_le _)
    /-
      🎉 no goals
    -/


lemma continuousMultilinearMapOption_apply_eq_self (B : G →L[𝕜] ContinuousMultilinearMap 𝕜 E F)
    (a : G) (v : Π i, E i) : B.continuousMultilinearMapOption (fun _ ↦ (a, v)) = B a v := rfl


