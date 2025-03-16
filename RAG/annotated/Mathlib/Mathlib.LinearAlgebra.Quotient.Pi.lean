/-- Lift a family of maps to the direct sum of quotients. -/
def piQuotientLift [Fintype ι] [DecidableEq ι] (p : ∀ i, Submodule R (Ms i)) (q : Submodule R N)
    (f : ∀ i, Ms i →ₗ[R] N) (hf : ∀ i, p i ≤ q.comap (f i)) : (∀ i, Ms i ⧸ p i) →ₗ[R] N ⧸ q :=
  lsum R (fun i => Ms i ⧸ p i) R fun i => (p i).mapQ q (f i) (hf i)


@[simp]
theorem piQuotientLift_mk [Fintype ι] [DecidableEq ι] (p : ∀ i, Submodule R (Ms i))
    (q : Submodule R N) (f : ∀ i, Ms i →ₗ[R] N) (hf : ∀ i, p i ≤ q.comap (f i)) (x : ∀ i, Ms i) :
    (piQuotientLift p q f hf fun i => Quotient.mk (x i)) = Quotient.mk (lsum _ _ R f x) := by
  /-
    ι : Type u_1
    R : Type u_2
    inst✝⁶ : CommRing R
    Ms : ι → Type u_3
    inst✝⁵ : (i : ι) → AddCommGroup (Ms i)
    inst✝⁴ : (i : ι) → Module R (Ms i)
    N : Type u_4
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    p : (i : ι) → Submodule R (Ms i)
    q : Submodule R N
    f : (i : ι) → LinearMap (RingHom.id R) (Ms i) N
    hf : ∀ (i : ι), LE.le (p i) (Submodule.comap (f i) q)
    x : (i : ι) → Ms i
    ⊢ Eq ((Submodule.piQuotientLift p q f hf) fun i => Submodule.Quotient.mk (x i) …
  -/
  rw [piQuotientLift, lsum_apply, sum_apply, ← mkQ_apply, lsum_apply, sum_apply, _root_.map_sum]
  /-
    ι : Type u_1
    R : Type u_2
    inst✝⁶ : CommRing R
    Ms : ι → Type u_3
    inst✝⁵ : (i : ι) → AddCommGroup (Ms i)
    inst✝⁴ : (i : ι) → Module R (Ms i)
    N : Type u_4
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    p : (i : ι) → Submodule R (Ms i)
    q : Submodule R N
    f : (i : ι) → LinearMap (RingHom.id R) (Ms i) N
    hf : ∀ (i : ι), LE.le (p i) (Submodule.comap (f i) q)
    x : (i : ι) → Ms i
    ⊢ Eq (Finset.univ.sum fun d => (((p d).mapQ q (f d) ⋯).comp (LinearMap.proj d) …
  -/
  simp only [coe_proj, mapQ_apply, mkQ_apply, comp_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem piQuotientLift_single [Fintype ι] [DecidableEq ι] (p : ∀ i, Submodule R (Ms i))
    (q : Submodule R N) (f : ∀ i, Ms i →ₗ[R] N) (hf : ∀ i, p i ≤ q.comap (f i)) (i)
    (x : Ms i ⧸ p i) : piQuotientLift p q f hf (Pi.single i x) = mapQ _ _ (f i) (hf i) x := by
  /-
    ι : Type u_1
    R : Type u_2
    inst✝⁶ : CommRing R
    Ms : ι → Type u_3
    inst✝⁵ : (i : ι) → AddCommGroup (Ms i)
    inst✝⁴ : (i : ι) → Module R (Ms i)
    N : Type u_4
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    p : (i : ι) → Submodule R (Ms i)
    q : Submodule R N
    f : (i : ι) → LinearMap (RingHom.id R) (Ms i) N
    hf : ∀ (i : ι), LE.le (p i) (Submodule.comap (f i) q)
    i : ι
    x : HasQuotient.Quotient (Ms i) (p i)
    ⊢ Eq ((Submodule.piQuotientLift p q f hf) (Pi.single i x)) (((p i).mapQ q (f i …
  -/
  simp_rw [piQuotientLift, lsum_apply, sum_apply, comp_apply, proj_apply]
  /-
    ι : Type u_1
    R : Type u_2
    inst✝⁶ : CommRing R
    Ms : ι → Type u_3
    inst✝⁵ : (i : ι) → AddCommGroup (Ms i)
    inst✝⁴ : (i : ι) → Module R (Ms i)
    N : Type u_4
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    p : (i : ι) → Submodule R (Ms i)
    q : Submodule R N
    f : (i : ι) → LinearMap (RingHom.id R) (Ms i) N
    hf : ∀ (i : ι), LE.le (p i) (Submodule.comap (f i) q)
    i : ι
    x : HasQuotient.Quotient (Ms i) (p i)
    ⊢ Eq (Finset.univ.sum fun x_1 => ((p x_1).mapQ q (f x_1) ⋯) (Pi.single i x x_1 …
  -/
  rw [Finset.sum_eq_single i]
    /-
      ι : Type u_1
      R : Type u_2
      inst✝⁶ : CommRing R
      Ms : ι → Type u_3
      inst✝⁵ : (i : ι) → AddCommGroup (Ms i)
      inst✝⁴ : (i : ι) → Module R (Ms i)
      N : Type u_4
      inst✝³ : AddCommGroup N
      inst✝² : Module R N
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      p : (i : ι) → Submodule R (Ms i)
      q : Submodule R N
      f : (i : ι) → LinearMap (RingHom.id R) (Ms i) N
      hf : ∀ (i : ι), LE.le (p i) (Submodule.comap (f i) q)
      i : ι
      x : HasQuotient.Quotient (Ms i) (p i)
      ⊢ Eq (((p i).mapQ q (f i) ⋯) (Pi.single i x i)) (((p i).mapQ q (f i) ⋯) x)
    -/
  · rw [Pi.single_eq_same]
    /-
      🎉 no goals
    -/
    /-
      case h₀
      ι : Type u_1
      R : Type u_2
      inst✝⁶ : CommRing R
      Ms : ι → Type u_3
      inst✝⁵ : (i : ι) → AddCommGroup (Ms i)
      inst✝⁴ : (i : ι) → Module R (Ms i)
      N : Type u_4
      inst✝³ : AddCommGroup N
      inst✝² : Module R N
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      p : (i : ι) → Submodule R (Ms i)
      q : Submodule R N
      f : (i : ι) → LinearMap (RingHom.id R) (Ms i) N
      hf : ∀ (i : ι), LE.le (p i) (Submodule.comap (f i) q)
      i : ι
      x : HasQuotient.Quotient (Ms i) (p i)
      ⊢ ∀ (b : ι), Membership.mem Finset.univ b → Ne b i → Eq (((p b).mapQ q (f b) ⋯ …
    -/
  · rintro j - hj
    /-
      case h₀
      ι : Type u_1
      R : Type u_2
      inst✝⁶ : CommRing R
      Ms : ι → Type u_3
      inst✝⁵ : (i : ι) → AddCommGroup (Ms i)
      inst✝⁴ : (i : ι) → Module R (Ms i)
      N : Type u_4
      inst✝³ : AddCommGroup N
      inst✝² : Module R N
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      p : (i : ι) → Submodule R (Ms i)
      q : Submodule R N
      f : (i : ι) → LinearMap (RingHom.id R) (Ms i) N
      hf : ∀ (i : ι), LE.le (p i) (Submodule.comap (f i) q)
      i : ι
      x : HasQuotient.Quotient (Ms i) (p i)
      j : ι
      hj : Ne j i
      ⊢ Eq (((p j).mapQ q (f j) ⋯) (Pi.single i x j)) 0
    -/
    rw [Pi.single_eq_of_ne hj, _root_.map_zero]
    /-
      🎉 no goals
    -/
    /-
      case h₁
      ι : Type u_1
      R : Type u_2
      inst✝⁶ : CommRing R
      Ms : ι → Type u_3
      inst✝⁵ : (i : ι) → AddCommGroup (Ms i)
      inst✝⁴ : (i : ι) → Module R (Ms i)
      N : Type u_4
      inst✝³ : AddCommGroup N
      inst✝² : Module R N
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      p : (i : ι) → Submodule R (Ms i)
      q : Submodule R N
      f : (i : ι) → LinearMap (RingHom.id R) (Ms i) N
      hf : ∀ (i : ι), LE.le (p i) (Submodule.comap (f i) q)
      i : ι
      x : HasQuotient.Quotient (Ms i) (p i)
      ⊢ Not (Membership.mem Finset.univ i) → Eq (((p i).mapQ q (f i) ⋯) (Pi.single i …
    -/
  · intros
    /-
      case h₁
      ι : Type u_1
      R : Type u_2
      inst✝⁶ : CommRing R
      Ms : ι → Type u_3
      inst✝⁵ : (i : ι) → AddCommGroup (Ms i)
      inst✝⁴ : (i : ι) → Module R (Ms i)
      N : Type u_4
      inst✝³ : AddCommGroup N
      inst✝² : Module R N
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      p : (i : ι) → Submodule R (Ms i)
      q : Submodule R N
      f : (i : ι) → LinearMap (RingHom.id R) (Ms i) N
      hf : ∀ (i : ι), LE.le (p i) (Submodule.comap (f i) q)
      i : ι
      x : HasQuotient.Quotient (Ms i) (p i)
      a✝ : Not (Membership.mem Finset.univ i)
      ⊢ Eq (((p i).mapQ q (f i) ⋯) (Pi.single i x i)) 0
    -/
    have := Finset.mem_univ i
    /-
      case h₁
      ι : Type u_1
      R : Type u_2
      inst✝⁶ : CommRing R
      Ms : ι → Type u_3
      inst✝⁵ : (i : ι) → AddCommGroup (Ms i)
      inst✝⁴ : (i : ι) → Module R (Ms i)
      N : Type u_4
      inst✝³ : AddCommGroup N
      inst✝² : Module R N
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      p : (i : ι) → Submodule R (Ms i)
      q : Submodule R N
      f : (i : ι) → LinearMap (RingHom.id R) (Ms i) N
      hf : ∀ (i : ι), LE.le (p i) (Submodule.comap (f i) q)
      i : ι
      x : HasQuotient.Quotient (Ms i) (p i)
      a✝ : Not (Membership.mem Finset.univ i)
      this : Membership.mem Finset.univ i
      ⊢ Eq (((p i).mapQ q (f i) ⋯) (Pi.single i x i)) 0
    -/
    contradiction
    /-
      🎉 no goals
    -/


/-- Lift a family of maps to a quotient of direct sums. -/
def quotientPiLift (p : ∀ i, Submodule R (Ms i)) (f : ∀ i, Ms i →ₗ[R] Ns i)
    (hf : ∀ i, p i ≤ ker (f i)) : (∀ i, Ms i) ⧸ pi Set.univ p →ₗ[R] ∀ i, Ns i :=
  (pi Set.univ p).liftQ (LinearMap.pi fun i => (f i).comp (proj i)) fun x hx =>
    mem_ker.mpr <| by
      /-
        ι : Type u_1
        R : Type u_2
        inst✝⁶ : CommRing R
        Ms : ι → Type u_3
        inst✝⁵ : (i : ι) → AddCommGroup (Ms i)
        inst✝⁴ : (i : ι) → Module R (Ms i)
        N : Type u_4
        inst✝³ : AddCommGroup N
        inst✝² : Module R N
        Ns : ι → Type u_5
        inst✝¹ : (i : ι) → AddCommGroup (Ns i)
        inst✝ : (i : ι) → Module R (Ns i)
        p : (i : ι) → Submodule R (Ms i)
        f : (i : ι) → LinearMap (RingHom.id R) (Ms i) (Ns i)
        hf : ∀ (i : ι), LE.le (p i) (LinearMap.ker (f i))
        x : (i : ι) → Ms i
        hx : Membership.mem (Submodule.pi Set.univ p) x
        ⊢ Eq ((LinearMap.pi fun i => (f i).comp (LinearMap.proj i)) x) 0
      -/
      ext i
      /-
        case h
        ι : Type u_1
        R : Type u_2
        inst✝⁶ : CommRing R
        Ms : ι → Type u_3
        inst✝⁵ : (i : ι) → AddCommGroup (Ms i)
        inst✝⁴ : (i : ι) → Module R (Ms i)
        N : Type u_4
        inst✝³ : AddCommGroup N
        inst✝² : Module R N
        Ns : ι → Type u_5
        inst✝¹ : (i : ι) → AddCommGroup (Ns i)
        inst✝ : (i : ι) → Module R (Ns i)
        p : (i : ι) → Submodule R (Ms i)
        f : (i : ι) → LinearMap (RingHom.id R) (Ms i) (Ns i)
        hf : ∀ (i : ι), LE.le (p i) (LinearMap.ker (f i))
        x : (i : ι) → Ms i
        hx : Membership.mem (Submodule.pi Set.univ p) x
        i : ι
        ⊢ Eq ((LinearMap.pi fun i => (f i).comp (LinearMap.proj i)) x i) (0 i)
      -/
      simpa using hf i (mem_pi.mp hx i (Set.mem_univ i))
      /-
        🎉 no goals
      -/


@[simp]
theorem quotientPiLift_mk (p : ∀ i, Submodule R (Ms i)) (f : ∀ i, Ms i →ₗ[R] Ns i)
    (hf : ∀ i, p i ≤ ker (f i)) (x : ∀ i, Ms i) :
    quotientPiLift p f hf (Quotient.mk x) = fun i => f i (x i) :=
  rfl


@[simp]
def toFun : ((∀ i, Ms i) ⧸ pi Set.univ p) → ∀ i, Ms i ⧸ p i :=
  quotientPiLift p (fun i => (p i).mkQ) fun i => (ker_mkQ (p i)).ge


theorem map_add (x y : ((i : ι) → Ms i) ⧸ pi Set.univ p) :
    toFun p (x + y) = toFun p x + toFun p y :=
  LinearMap.map_add (quotientPiLift p (fun i => (p i).mkQ) fun i => (ker_mkQ (p i)).ge) x y


theorem map_smul (r : R) (x : ((i : ι) → Ms i) ⧸ pi Set.univ p) :
    toFun p (r • x) = (RingHom.id R r) • toFun p x :=
  LinearMap.map_smul (quotientPiLift p (fun i => (p i).mkQ) fun i => (ker_mkQ (p i)).ge) r x


@[simp]
def invFun : (∀ i, Ms i ⧸ p i) → (∀ i, Ms i) ⧸ pi Set.univ p :=
  piQuotientLift p (pi Set.univ p) _ fun _ => le_comap_single_pi p


theorem left_inv : Function.LeftInverse (invFun p) (toFun p) := fun x =>
  Submodule.Quotient.induction_on _ x fun x' => by
    /-
      ι : Type u_1
      R : Type u_2
      inst✝⁴ : CommRing R
      Ms : ι → Type u_3
      inst✝³ : (i : ι) → AddCommGroup (Ms i)
      inst✝² : (i : ι) → Module R (Ms i)
      p : (i : ι) → Submodule R (Ms i)
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      x : HasQuotient.Quotient ((i : ι) → Ms i) (Submodule.pi Set.univ p)
      x' : (i : ι) → Ms i
      ⊢ Eq (Submodule.quotientPi_aux.invFun p (Submodule.quotientPi_aux.toFun p (Sub …
    -/
    dsimp only [toFun, invFun]
    rw [quotientPiLift_mk p, funext fun i => (mkQ_apply (p i) (x' i)), piQuotientLift_mk p,
      lsum_single, id_apply]


theorem right_inv : Function.RightInverse (invFun p) (toFun p) := by
  /-
    ι : Type u_1
    R : Type u_2
    inst✝⁴ : CommRing R
    Ms : ι → Type u_3
    inst✝³ : (i : ι) → AddCommGroup (Ms i)
    inst✝² : (i : ι) → Module R (Ms i)
    p : (i : ι) → Submodule R (Ms i)
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    ⊢ Function.RightInverse (Submodule.quotientPi_aux.invFun p) (Submodule.quotien …
  -/
  dsimp only [toFun, invFun]
  /-
    ι : Type u_1
    R : Type u_2
    inst✝⁴ : CommRing R
    Ms : ι → Type u_3
    inst✝³ : (i : ι) → AddCommGroup (Ms i)
    inst✝² : (i : ι) → Module R (Ms i)
    p : (i : ι) → Submodule R (Ms i)
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    ⊢ Function.RightInverse ⇑(Submodule.piQuotientLift p (Submodule.pi Set.univ p) …
  -/
  rw [Function.rightInverse_iff_comp, ← coe_comp, ← @id_coe R]
  refine congr_arg _ (pi_ext fun i x => Submodule.Quotient.induction_on _ x fun x' =>
    funext fun j => ?_)
  rw [comp_apply, piQuotientLift_single, mapQ_apply,
    quotientPiLift_mk, id_apply]
  /-
    ι : Type u_1
    R : Type u_2
    inst✝⁴ : CommRing R
    Ms : ι → Type u_3
    inst✝³ : (i : ι) → AddCommGroup (Ms i)
    inst✝² : (i : ι) → Module R (Ms i)
    p : (i : ι) → Submodule R (Ms i)
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    i : ι
    x : HasQuotient.Quotient (Ms i) (p i)
    x' : Ms i
    j : ι
    ⊢ Eq ((fun i_1 => (p i_1).mkQ ((LinearMap.single R Ms i) x' i_1)) j) (Pi.singl …
  -/
  by_cases hij : i = j <;> simp only [mkQ_apply, coe_single]
    /-
      case pos
      ι : Type u_1
      R : Type u_2
      inst✝⁴ : CommRing R
      Ms : ι → Type u_3
      inst✝³ : (i : ι) → AddCommGroup (Ms i)
      inst✝² : (i : ι) → Module R (Ms i)
      p : (i : ι) → Submodule R (Ms i)
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      i : ι
      x : HasQuotient.Quotient (Ms i) (p i)
      x' : Ms i
      j : ι
      hij : Eq i j
      ⊢ Eq (Submodule.Quotient.mk (Pi.single i x' j)) (Pi.single i (Submodule.Quotie …
    -/
  · subst hij
    /-
      case pos
      ι : Type u_1
      R : Type u_2
      inst✝⁴ : CommRing R
      Ms : ι → Type u_3
      inst✝³ : (i : ι) → AddCommGroup (Ms i)
      inst✝² : (i : ι) → Module R (Ms i)
      p : (i : ι) → Submodule R (Ms i)
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      i : ι
      x : HasQuotient.Quotient (Ms i) (p i)
      x' : Ms i
      ⊢ Eq (Submodule.Quotient.mk (Pi.single i x' i)) (Pi.single i (Submodule.Quotie …
    -/
    rw [Pi.single_eq_same, Pi.single_eq_same]
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u_1
      R : Type u_2
      inst✝⁴ : CommRing R
      Ms : ι → Type u_3
      inst✝³ : (i : ι) → AddCommGroup (Ms i)
      inst✝² : (i : ι) → Module R (Ms i)
      p : (i : ι) → Submodule R (Ms i)
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      i : ι
      x : HasQuotient.Quotient (Ms i) (p i)
      x' : Ms i
      j : ι
      hij : Not (Eq i j)
      ⊢ Eq (Submodule.Quotient.mk (Pi.single i x' j)) (Pi.single i (Submodule.Quotie …
    -/
  · rw [Pi.single_eq_of_ne (Ne.symm hij), Pi.single_eq_of_ne (Ne.symm hij), Quotient.mk_zero]
    /-
      🎉 no goals
    -/


open quotientPi_aux in
/-- The quotient of a direct sum is the direct sum of quotients. -/
@[simps!]
def quotientPi [Fintype ι] [DecidableEq ι] (p : ∀ i, Submodule R (Ms i)) :
    ((∀ i, Ms i) ⧸ pi Set.univ p) ≃ₗ[R] ∀ i, Ms i ⧸ p i where
  toFun := toFun p
  invFun := invFun p
  map_add' := map_add p
  map_smul' := quotientPi_aux.map_smul p
  left_inv := left_inv p
  right_inv := right_inv p


