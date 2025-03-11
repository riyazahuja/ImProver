/-- A formal multilinear series over a field `𝕜`, from `E` to `F`, is given by a family of
multilinear maps from `E^n` to `F` for all `n`. -/
@[nolint unusedArguments]
def FormalMultilinearSeries (𝕜 : Type*) (E : Type*) (F : Type*) [Ring 𝕜] [AddCommGroup E]
    [Module 𝕜 E] [TopologicalSpace E] [TopologicalAddGroup E] [ContinuousConstSMul 𝕜 E]
    [AddCommGroup F] [Module 𝕜 F] [TopologicalSpace F] [TopologicalAddGroup F]
    [ContinuousConstSMul 𝕜 F] :=
  ∀ n : ℕ, E[×n]→L[𝕜] F

-- Porting note: was `deriving`

instance : AddCommGroup (FormalMultilinearSeries 𝕜 E F) :=
  inferInstanceAs <| AddCommGroup <| ∀ n : ℕ, E[×n]→L[𝕜] F


instance : Inhabited (FormalMultilinearSeries 𝕜 E F) :=
  ⟨0⟩


instance (𝕜') [Semiring 𝕜'] [Module 𝕜' F] [ContinuousConstSMul 𝕜' F] [SMulCommClass 𝕜 𝕜' F] :
    Module 𝕜' (FormalMultilinearSeries 𝕜 E F) :=
  inferInstanceAs <| Module 𝕜' <| ∀ n : ℕ, E[×n]→L[𝕜] F


@[simp, nolint simpNF]
theorem zero_apply (n : ℕ) : (0 : FormalMultilinearSeries 𝕜 E F) n = 0 := rfl


@[simp]
theorem neg_apply (f : FormalMultilinearSeries 𝕜 E F) (n : ℕ) : (-f) n = - f n := rfl


@[simp]
theorem add_apply (p q : FormalMultilinearSeries 𝕜 E F) (n : ℕ) : (p + q) n = p n + q n := rfl


@[simp]
theorem sub_apply (p q : FormalMultilinearSeries 𝕜 E F) (n : ℕ) : (p - q) n = p n - q n := rfl


@[ext]
protected theorem ext {p q : FormalMultilinearSeries 𝕜 E F} (h : ∀ n, p n = q n) : p = q :=
  funext h


protected theorem ne_iff {p q : FormalMultilinearSeries 𝕜 E F} : p ≠ q ↔ ∃ n, p n ≠ q n :=
  Function.ne_iff


/-- Cartesian product of two formal multilinear series (with the same field `𝕜` and the same source
space, but possibly different target spaces). -/
def prod (p : FormalMultilinearSeries 𝕜 E F) (q : FormalMultilinearSeries 𝕜 E G) :
    FormalMultilinearSeries 𝕜 E (F × G)
  | n => (p n).prod (q n)


/-- Product of formal multilinear series (with the same field `𝕜` and the same source
space, but possibly different target spaces). -/
@[simp] def pi {ι : Type*} {F : ι → Type*}
    [∀ i, AddCommGroup (F i)] [∀ i, Module 𝕜 (F i)] [∀ i, TopologicalSpace (F i)]
    [∀ i, TopologicalAddGroup (F i)] [∀ i, ContinuousConstSMul 𝕜 (F i)]
    (p : Π i, FormalMultilinearSeries 𝕜 E (F i)) :
    FormalMultilinearSeries 𝕜 E (Π i, F i)
  | n => ContinuousMultilinearMap.pi (fun i ↦ p i n)


/-- Killing the zeroth coefficient in a formal multilinear series -/
def removeZero (p : FormalMultilinearSeries 𝕜 E F) : FormalMultilinearSeries 𝕜 E F
  | 0 => 0
  | n + 1 => p (n + 1)


@[simp]
theorem removeZero_coeff_zero (p : FormalMultilinearSeries 𝕜 E F) : p.removeZero 0 = 0 :=
  rfl


@[simp]
theorem removeZero_coeff_succ (p : FormalMultilinearSeries 𝕜 E F) (n : ℕ) :
    p.removeZero (n + 1) = p (n + 1) :=
  rfl


theorem removeZero_of_pos (p : FormalMultilinearSeries 𝕜 E F) {n : ℕ} (h : 0 < n) :
    p.removeZero n = p n := by
  /-
    𝕜 : Type u
    E : Type v
    F : Type w
    inst✝¹⁰ : Ring 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : Module 𝕜 E
    inst✝⁷ : TopologicalSpace E
    inst✝⁶ : TopologicalAddGroup E
    inst✝⁵ : ContinuousConstSMul 𝕜 E
    inst✝⁴ : AddCommGroup F
    inst✝³ : Module 𝕜 F
    inst✝² : TopologicalSpace F
    inst✝¹ : TopologicalAddGroup F
    inst✝ : ContinuousConstSMul 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    h : LT.lt 0 n
    ⊢ Eq (p.removeZero n) (p n)
  -/
  rw [← Nat.succ_pred_eq_of_pos h]
  /-
    𝕜 : Type u
    E : Type v
    F : Type w
    inst✝¹⁰ : Ring 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : Module 𝕜 E
    inst✝⁷ : TopologicalSpace E
    inst✝⁶ : TopologicalAddGroup E
    inst✝⁵ : ContinuousConstSMul 𝕜 E
    inst✝⁴ : AddCommGroup F
    inst✝³ : Module 𝕜 F
    inst✝² : TopologicalSpace F
    inst✝¹ : TopologicalAddGroup F
    inst✝ : ContinuousConstSMul 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    h : LT.lt 0 n
    ⊢ Eq (p.removeZero n.pred.succ) (p n.pred.succ)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Convenience congruence lemma stating in a dependent setting that, if the arguments to a formal
multilinear series are equal, then the values are also equal. -/
theorem congr (p : FormalMultilinearSeries 𝕜 E F) {m n : ℕ} {v : Fin m → E} {w : Fin n → E}
    (h1 : m = n) (h2 : ∀ (i : ℕ) (him : i < m) (hin : i < n), v ⟨i, him⟩ = w ⟨i, hin⟩) :
    p m v = p n w := by
  /-
    𝕜 : Type u
    E : Type v
    F : Type w
    inst✝¹⁰ : Ring 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : Module 𝕜 E
    inst✝⁷ : TopologicalSpace E
    inst✝⁶ : TopologicalAddGroup E
    inst✝⁵ : ContinuousConstSMul 𝕜 E
    inst✝⁴ : AddCommGroup F
    inst✝³ : Module 𝕜 F
    inst✝² : TopologicalSpace F
    inst✝¹ : TopologicalAddGroup F
    inst✝ : ContinuousConstSMul 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    m n : Nat
    v : Fin m → E
    w : Fin n → E
    h1 : Eq m n
    h2 : ∀ (i : Nat) (him : LT.lt i m) (hin : LT.lt i n), Eq (v ⟨i, him⟩) (w ⟨i, h …
    ⊢ Eq ((p m) v) ((p n) w)
  -/
  subst n
  /-
    𝕜 : Type u
    E : Type v
    F : Type w
    inst✝¹⁰ : Ring 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : Module 𝕜 E
    inst✝⁷ : TopologicalSpace E
    inst✝⁶ : TopologicalAddGroup E
    inst✝⁵ : ContinuousConstSMul 𝕜 E
    inst✝⁴ : AddCommGroup F
    inst✝³ : Module 𝕜 F
    inst✝² : TopologicalSpace F
    inst✝¹ : TopologicalAddGroup F
    inst✝ : ContinuousConstSMul 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    m : Nat
    v w : Fin m → E
    h2 : ∀ (i : Nat) (him hin : LT.lt i m), Eq (v ⟨i, him⟩) (w ⟨i, hin⟩)
    ⊢ Eq ((p m) v) ((p m) w)
  -/
  congr with ⟨i, hi⟩
  /-
    case h.e_6.h.h.mk
    𝕜 : Type u
    E : Type v
    F : Type w
    inst✝¹⁰ : Ring 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : Module 𝕜 E
    inst✝⁷ : TopologicalSpace E
    inst✝⁶ : TopologicalAddGroup E
    inst✝⁵ : ContinuousConstSMul 𝕜 E
    inst✝⁴ : AddCommGroup F
    inst✝³ : Module 𝕜 F
    inst✝² : TopologicalSpace F
    inst✝¹ : TopologicalAddGroup F
    inst✝ : ContinuousConstSMul 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    m : Nat
    v w : Fin m → E
    h2 : ∀ (i : Nat) (him hin : LT.lt i m), Eq (v ⟨i, him⟩) (w ⟨i, hin⟩)
    i : Nat
    hi : LT.lt i m
    ⊢ Eq (v ⟨i, hi⟩) (w ⟨i, hi⟩)
  -/
  exact h2 i hi hi
  /-
    🎉 no goals
  -/


lemma congr_zero (p : FormalMultilinearSeries 𝕜 E F) {k l : ℕ} (h : k = l) (h' : p k = 0) :
    p l = 0 := by
  /-
    𝕜 : Type u
    E : Type v
    F : Type w
    inst✝¹⁰ : Ring 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : Module 𝕜 E
    inst✝⁷ : TopologicalSpace E
    inst✝⁶ : TopologicalAddGroup E
    inst✝⁵ : ContinuousConstSMul 𝕜 E
    inst✝⁴ : AddCommGroup F
    inst✝³ : Module 𝕜 F
    inst✝² : TopologicalSpace F
    inst✝¹ : TopologicalAddGroup F
    inst✝ : ContinuousConstSMul 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    k l : Nat
    h : Eq k l
    h' : Eq (p k) 0
    ⊢ Eq (p l) 0
  -/
  subst h; exact h'
           /-
             🎉 no goals
           -/


/-- Composing each term `pₙ` in a formal multilinear series with `(u, ..., u)` where `u` is a fixed
continuous linear map, gives a new formal multilinear series `p.compContinuousLinearMap u`. -/
def compContinuousLinearMap (p : FormalMultilinearSeries 𝕜 F G) (u : E →L[𝕜] F) :
    FormalMultilinearSeries 𝕜 E G := fun n => (p n).compContinuousLinearMap fun _ : Fin n => u


@[simp]
theorem compContinuousLinearMap_apply (p : FormalMultilinearSeries 𝕜 F G) (u : E →L[𝕜] F) (n : ℕ)
    (v : Fin n → E) : (p.compContinuousLinearMap u) n v = p n (u ∘ v) :=
  rfl


/-- Reinterpret a formal `𝕜'`-multilinear series as a formal `𝕜`-multilinear series. -/
@[simp]
protected def restrictScalars (p : FormalMultilinearSeries 𝕜' E F) :
    FormalMultilinearSeries 𝕜 E F := fun n => (p n).restrictScalars 𝕜


/-- Forgetting the zeroth term in a formal multilinear series, and interpreting the following terms
as multilinear maps into `E →L[𝕜] F`. If `p` is the Taylor series (`HasFTaylorSeriesUpTo`) of a
function, then `p.shift` is the Taylor series of the derivative of the function. Note that the
`p.sum` of a Taylor series `p` does not give the original function; for a formal multilinear
series that sums to the derivative of `p.sum`, see `HasFPowerSeriesOnBall.fderiv`. -/
def shift : FormalMultilinearSeries 𝕜 E (E →L[𝕜] F) := fun n => (p n.succ).curryRight


/-- Adding a zeroth term to a formal multilinear series taking values in `E →L[𝕜] F`. This
corresponds to starting from a Taylor series (`HasFTaylorSeriesUpTo`) for the derivative of a
function, and building a Taylor series for the function itself. -/
def unshift (q : FormalMultilinearSeries 𝕜 E (E →L[𝕜] F)) (z : F) : FormalMultilinearSeries 𝕜 E F
  | 0 => (continuousMultilinearCurryFin0 𝕜 E F).symm z
  | n + 1 => (continuousMultilinearCurryRightEquiv' 𝕜 n E F).symm (q n)


theorem unshift_shift {p : FormalMultilinearSeries 𝕜 E (E →L[𝕜] F)} {z : F} :
    (p.unshift z).shift = p := by
  /-
    𝕜 : Type u
    E : Type v
    F : Type w
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E (ContinuousLinearMap (RingHom.id 𝕜) E F)
    z : F
    ⊢ Eq (p.unshift z).shift p
  -/
  ext1 n
  /-
    case h
    𝕜 : Type u
    E : Type v
    F : Type w
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E (ContinuousLinearMap (RingHom.id 𝕜) E F)
    z : F
    n : Nat
    ⊢ Eq ((p.unshift z).shift n) (p n)
  -/
  simp [shift, unshift]
  /-
    case h
    𝕜 : Type u
    E : Type v
    F : Type w
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E (ContinuousLinearMap (RingHom.id 𝕜) E F)
    z : F
    n : Nat
    ⊢ Eq ((continuousMultilinearCurryRightEquiv' 𝕜 n E F).symm (p n)).curryRight ( …
  -/
  exact LinearIsometryEquiv.apply_symm_apply (continuousMultilinearCurryRightEquiv' 𝕜 n E F) (p n)
  /-
    🎉 no goals
  -/


/-- Composing each term `pₙ` in a formal multilinear series with a continuous linear map `f` on the
left gives a new formal multilinear series `f.compFormalMultilinearSeries p` whose general term
is `f ∘ pₙ`. -/
def compFormalMultilinearSeries (f : F →L[𝕜] G) (p : FormalMultilinearSeries 𝕜 E F) :
    FormalMultilinearSeries 𝕜 E G := fun n => f.compContinuousMultilinearMap (p n)


@[simp]
theorem compFormalMultilinearSeries_apply (f : F →L[𝕜] G) (p : FormalMultilinearSeries 𝕜 E F)
    (n : ℕ) : (f.compFormalMultilinearSeries p) n = f.compContinuousMultilinearMap (p n) :=
  rfl


theorem compFormalMultilinearSeries_apply' (f : F →L[𝕜] G) (p : FormalMultilinearSeries 𝕜 E F)
    (n : ℕ) (v : Fin n → E) : (f.compFormalMultilinearSeries p) n v = f (p n v) :=
  rfl


/-- Realize a ContinuousMultilinearMap on `∀ i : ι, E i` as the evaluation of a
FormalMultilinearSeries by choosing an arbitrary identification `ι ≃ Fin (Fintype.card ι)`. -/
noncomputable def toFormalMultilinearSeries : FormalMultilinearSeries 𝕜 (∀ i, E i) F :=
  fun n ↦ if h : Fintype.card ι = n then
    (f.compContinuousLinearMap .proj).domDomCongr (Fintype.equivFinOfCardEq h)
  else 0


/-- The index of the first non-zero coefficient in `p` (or `0` if all coefficients are zero). This
  is the order of the isolated zero of an analytic function `f` at a point if `p` is the Taylor
  series of `f` at that point. -/
noncomputable def order (p : FormalMultilinearSeries 𝕜 E F) : ℕ :=
  sInf { n | p n ≠ 0 }


@[simp]
                                                                         /-
                                                                           𝕜 : Type u
                                                                           E : Type v
                                                                           F : Type w
                                                                           inst✝¹⁰ : Ring 𝕜
                                                                           inst✝⁹ : AddCommGroup E
                                                                           inst✝⁸ : Module 𝕜 E
                                                                           inst✝⁷ : TopologicalSpace E
                                                                           inst✝⁶ : TopologicalAddGroup E
                                                                           inst✝⁵ : ContinuousConstSMul 𝕜 E
                                                                           inst✝⁴ : AddCommGroup F
                                                                           inst✝³ : Module 𝕜 F
                                                                           inst✝² : TopologicalSpace F
                                                                           inst✝¹ : TopologicalAddGroup F
                                                                           inst✝ : ContinuousConstSMul 𝕜 F
                                                                           ⊢ Eq (FormalMultilinearSeries.order 0) 0
                                                                         -/
theorem order_zero : (0 : FormalMultilinearSeries 𝕜 E F).order = 0 := by simp [order]
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


                                                                           /-
                                                                             𝕜 : Type u
                                                                             E : Type v
                                                                             F : Type w
                                                                             inst✝¹⁰ : Ring 𝕜
                                                                             inst✝⁹ : AddCommGroup E
                                                                             inst✝⁸ : Module 𝕜 E
                                                                             inst✝⁷ : TopologicalSpace E
                                                                             inst✝⁶ : TopologicalAddGroup E
                                                                             inst✝⁵ : ContinuousConstSMul 𝕜 E
                                                                             inst✝⁴ : AddCommGroup F
                                                                             inst✝³ : Module 𝕜 F
                                                                             inst✝² : TopologicalSpace F
                                                                             inst✝¹ : TopologicalAddGroup F
                                                                             inst✝ : ContinuousConstSMul 𝕜 F
                                                                             p : FormalMultilinearSeries 𝕜 E F
                                                                             hp : Ne p.order 0
                                                                             h : Eq p 0
                                                                             ⊢ False
                                                                           -/
theorem ne_zero_of_order_ne_zero (hp : p.order ≠ 0) : p ≠ 0 := fun h => by simp [h] at hp
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


theorem order_eq_find [DecidablePred fun n => p n ≠ 0] (hp : ∃ n, p n ≠ 0) :
                                /-
                                  𝕜 : Type u
                                  E : Type v
                                  F : Type w
                                  inst✝¹¹ : Ring 𝕜
                                  inst✝¹⁰ : AddCommGroup E
                                  inst✝⁹ : Module 𝕜 E
                                  inst✝⁸ : TopologicalSpace E
                                  inst✝⁷ : TopologicalAddGroup E
                                  inst✝⁶ : ContinuousConstSMul 𝕜 E
                                  inst✝⁵ : AddCommGroup F
                                  inst✝⁴ : Module 𝕜 F
                                  inst✝³ : TopologicalSpace F
                                  inst✝² : TopologicalAddGroup F
                                  inst✝¹ : ContinuousConstSMul 𝕜 F
                                  p : FormalMultilinearSeries 𝕜 E F
                                  inst✝ : DecidablePred fun n => Ne (p n) 0
                                  hp : Exists fun n => Ne (p n) 0
                                  ⊢ Eq p.order (Nat.find hp)
                                -/
    p.order = Nat.find hp := by convert Nat.sInf_def hp
                                /-
                                  🎉 no goals
                                -/


theorem order_eq_find' [DecidablePred fun n => p n ≠ 0] (hp : p ≠ 0) :
    p.order = Nat.find (FormalMultilinearSeries.ne_iff.mp hp) :=
  order_eq_find _


theorem order_eq_zero_iff' : p.order = 0 ↔ p = 0 ∨ p 0 ≠ 0 := by
  simpa [order, Nat.sInf_eq_zero, FormalMultilinearSeries.ext_iff, eq_empty_iff_forall_not_mem]
    using or_comm


theorem order_eq_zero_iff (hp : p ≠ 0) : p.order = 0 ↔ p 0 ≠ 0 := by
  /-
    𝕜 : Type u
    E : Type v
    F : Type w
    inst✝¹⁰ : Ring 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : Module 𝕜 E
    inst✝⁷ : TopologicalSpace E
    inst✝⁶ : TopologicalAddGroup E
    inst✝⁵ : ContinuousConstSMul 𝕜 E
    inst✝⁴ : AddCommGroup F
    inst✝³ : Module 𝕜 F
    inst✝² : TopologicalSpace F
    inst✝¹ : TopologicalAddGroup F
    inst✝ : ContinuousConstSMul 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    hp : Ne p 0
    ⊢ Iff (Eq p.order 0) (Ne (p 0) 0)
  -/
  simp [order_eq_zero_iff', hp]
  /-
    🎉 no goals
  -/


theorem apply_order_ne_zero (hp : p ≠ 0) : p p.order ≠ 0 :=
  Nat.sInf_mem (FormalMultilinearSeries.ne_iff.1 hp)


theorem apply_order_ne_zero' (hp : p.order ≠ 0) : p p.order ≠ 0 :=
  apply_order_ne_zero (ne_zero_of_order_ne_zero hp)


theorem apply_eq_zero_of_lt_order (hp : n < p.order) : p n = 0 :=
  by_contra <| Nat.not_mem_of_lt_sInf hp


/-- The `n`th coefficient of `p` when seen as a power series. -/
def coeff (p : FormalMultilinearSeries 𝕜 𝕜 E) (n : ℕ) : E :=
  p n 1


theorem mkPiRing_coeff_eq (p : FormalMultilinearSeries 𝕜 𝕜 E) (n : ℕ) :
    ContinuousMultilinearMap.mkPiRing 𝕜 (Fin n) (p.coeff n) = p n :=
  (p n).mkPiRing_apply_one_eq_self


@[simp]
theorem apply_eq_prod_smul_coeff : p n y = (∏ i, y i) • p.coeff n := by
  /-
    𝕜 : Type u
    E : Type v
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    p : FormalMultilinearSeries 𝕜 𝕜 E
    n : Nat
    y : Fin n → 𝕜
    ⊢ Eq ((p n) y) (HSMul.hSMul (Finset.univ.prod fun i => y i) (p.coeff n))
  -/
  convert (p n).toMultilinearMap.map_smul_univ y 1
  /-
    case h.e'_2.h.e'_1.h
    𝕜 : Type u
    E : Type v
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    p : FormalMultilinearSeries 𝕜 𝕜 E
    n : Nat
    y : Fin n → 𝕜
    x✝ : Fin n
    ⊢ Eq (y x✝) (HSMul.hSMul (y x✝) (1 x✝))
  -/
  simp only [Pi.one_apply, Algebra.id.smul_eq_mul, mul_one]
  /-
    🎉 no goals
  -/


theorem coeff_eq_zero : p.coeff n = 0 ↔ p n = 0 := by
  /-
    𝕜 : Type u
    E : Type v
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    p : FormalMultilinearSeries 𝕜 𝕜 E
    n : Nat
    ⊢ Iff (Eq (p.coeff n) 0) (Eq (p n) 0)
  -/
  rw [← mkPiRing_coeff_eq p, ContinuousMultilinearMap.mkPiRing_eq_zero_iff]
  /-
    🎉 no goals
  -/


                                                                             /-
                                                                               𝕜 : Type u
                                                                               E : Type v
                                                                               inst✝² : NontriviallyNormedField 𝕜
                                                                               inst✝¹ : NormedAddCommGroup E
                                                                               inst✝ : NormedSpace 𝕜 E
                                                                               p : FormalMultilinearSeries 𝕜 𝕜 E
                                                                               n : Nat
                                                                               z : 𝕜
                                                                               ⊢ Eq ((p n) fun x => z) (HSMul.hSMul (HPow.hPow z n) (p.coeff n))
                                                                             -/
theorem apply_eq_pow_smul_coeff : (p n fun _ => z) = z ^ n • p.coeff n := by simp
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


@[simp]
theorem norm_apply_eq_norm_coef : ‖p n‖ = ‖coeff p n‖ := by
  /-
    𝕜 : Type u
    E : Type v
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    p : FormalMultilinearSeries 𝕜 𝕜 E
    n : Nat
    ⊢ Eq (Norm.norm (p n)) (Norm.norm (p.coeff n))
  -/
  rw [← mkPiRing_coeff_eq p, ContinuousMultilinearMap.norm_mkPiRing]
  /-
    🎉 no goals
  -/


/-- The formal counterpart of `dslope`, corresponding to the expansion of `(f z - f 0) / z`. If `f`
has `p` as a power series, then `dslope f` has `fslope p` as a power series. -/
noncomputable def fslope (p : FormalMultilinearSeries 𝕜 𝕜 E) : FormalMultilinearSeries 𝕜 𝕜 E :=
  fun n => (p (n + 1)).curryLeft 1


@[simp]
theorem coeff_fslope : p.fslope.coeff n = p.coeff (n + 1) := by
  /-
    𝕜 : Type u
    E : Type v
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    p : FormalMultilinearSeries 𝕜 𝕜 E
    n : Nat
    ⊢ Eq (p.fslope.coeff n) (p.coeff (HAdd.hAdd n 1))
  -/
  simp only [fslope, coeff, ContinuousMultilinearMap.curryLeft_apply]
  /-
    𝕜 : Type u
    E : Type v
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    p : FormalMultilinearSeries 𝕜 𝕜 E
    n : Nat
    ⊢ Eq ((p (HAdd.hAdd n 1)) (Fin.cons 1 1)) ((p (HAdd.hAdd n 1)) 1)
  -/
  congr 1
  /-
    case h.e_6.h
    𝕜 : Type u
    E : Type v
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    p : FormalMultilinearSeries 𝕜 𝕜 E
    n : Nat
    ⊢ Eq (Fin.cons 1 1) 1
  -/
  exact Fin.cons_self_tail (fun _ => (1 : 𝕜))
  /-
    🎉 no goals
  -/


@[simp]
theorem coeff_iterate_fslope (k n : ℕ) : (fslope^[k] p).coeff n = p.coeff (n + k) := by
  induction k generalizing p with
  | zero => rfl
  | succ k ih => simp [ih, add_assoc]


/-- The formal multilinear series where all terms of positive degree are equal to zero, and the term
of degree zero is `c`. It is the power series expansion of the constant function equal to `c`
everywhere. -/
def constFormalMultilinearSeries (𝕜 : Type*) [NontriviallyNormedField 𝕜] (E : Type*)
    [NormedAddCommGroup E] [NormedSpace 𝕜 E] [ContinuousConstSMul 𝕜 E] [TopologicalAddGroup E]
    {F : Type*} [NormedAddCommGroup F] [TopologicalAddGroup F] [NormedSpace 𝕜 F]
    [ContinuousConstSMul 𝕜 F] (c : F) : FormalMultilinearSeries 𝕜 E F
  | 0 => ContinuousMultilinearMap.uncurry0 _ _ c
  | _ => 0


@[simp]
theorem constFormalMultilinearSeries_apply [NontriviallyNormedField 𝕜] [NormedAddCommGroup E]
    [NormedAddCommGroup F] [NormedSpace 𝕜 E] [NormedSpace 𝕜 F] {c : F} {n : ℕ} (hn : n ≠ 0) :
    constFormalMultilinearSeries 𝕜 E c n = 0 :=
  Nat.casesOn n (fun hn => (hn rfl).elim) (fun _ _ => rfl) hn


@[simp]
lemma constFormalMultilinearSeries_zero [NontriviallyNormedField 𝕜] [NormedAddCommGroup E ]
    [NormedAddCommGroup F] [NormedSpace 𝕜 E] [NormedSpace 𝕜 F] :
    constFormalMultilinearSeries 𝕜 E (0 : F) = 0 := by
  /-
    𝕜 : Type u
    E : Type v
    F : Type w
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : NormedSpace 𝕜 F
    ⊢ Eq (constFormalMultilinearSeries 𝕜 E 0) 0
  -/
  ext n x
  simp only [FormalMultilinearSeries.zero_apply, ContinuousMultilinearMap.zero_apply,
    constFormalMultilinearSeries]
  /-
    case h.H
    𝕜 : Type u
    E : Type v
    F : Type w
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : NormedSpace 𝕜 F
    n : Nat
    x : Fin n → E
    ⊢ Eq ((constFormalMultilinearSeries.match_1 (fun x => ContinuousMultilinearMap …
  -/
  induction n
    /-
      case h.H.zero
      𝕜 : Type u
      E : Type v
      F : Type w
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : NormedSpace 𝕜 F
      x : Fin 0 → E
      ⊢ Eq ((constFormalMultilinearSeries.match_1 (fun x => ContinuousMultilinearMap …
    -/
  · simp only [ContinuousMultilinearMap.uncurry0_apply]
    /-
      🎉 no goals
    -/
    /-
      case h.H.succ
      𝕜 : Type u
      E : Type v
      F : Type w
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : NormedSpace 𝕜 F
      n✝ : Nat
      a✝ : ∀ (x : Fin n✝ → E), Eq ((constFormalMultilinearSeries.match_1 (fun x => C …
      x : Fin (HAdd.hAdd n✝ 1) → E
      ⊢ Eq ((constFormalMultilinearSeries.match_1 (fun x => ContinuousMultilinearMap …
    -/
  · simp only [constFormalMultilinearSeries.match_1.eq_2, ContinuousMultilinearMap.zero_apply]
    /-
      🎉 no goals
    -/


/-- Formal power series of a continuous linear map `f : E →L[𝕜] F` at `x : E`:
`f y = f x + f (y - x)`. -/
def fpowerSeries (f : E →L[𝕜] F) (x : E) : FormalMultilinearSeries 𝕜 E F
  | 0 => ContinuousMultilinearMap.uncurry0 𝕜 _ (f x)
  | 1 => (continuousMultilinearCurryFin1 𝕜 E F).symm f
  | _ => 0


@[simp]
theorem fpowerSeries_apply_zero (f : E →L[𝕜] F) (x : E) :
    f.fpowerSeries x 0 = ContinuousMultilinearMap.uncurry0 𝕜 _ (f x) :=
  rfl


@[simp]
theorem fpowerSeries_apply_one (f : E →L[𝕜] F) (x : E) :
    f.fpowerSeries x 1 = (continuousMultilinearCurryFin1 𝕜 E F).symm f :=
  rfl


@[simp]
theorem fpowerSeries_apply_add_two (f : E →L[𝕜] F) (x : E) (n : ℕ) : f.fpowerSeries x (n + 2) = 0 :=
  rfl


