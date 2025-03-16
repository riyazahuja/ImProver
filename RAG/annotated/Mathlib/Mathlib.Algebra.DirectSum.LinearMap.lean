/-- If a linear map `f : M₁ → M₂` respects direct sum decompositions of `M₁` and `M₂`, then it has a
block diagonal matrix with respect to bases compatible with the direct sum decompositions. -/
lemma toMatrix_directSum_collectedBasis_eq_blockDiagonal' {R M₁ M₂ : Type*} [CommSemiring R]
    [AddCommMonoid M₁] [Module R M₁] {N₁ : ι → Submodule R M₁} (h₁ : IsInternal N₁)
    [AddCommMonoid M₂] [Module R M₂] {N₂ : ι → Submodule R M₂} (h₂ : IsInternal N₂)
    {κ₁ κ₂ : ι → Type*} [∀ i, Fintype (κ₁ i)] [∀ i, Finite (κ₂ i)] [∀ i, DecidableEq (κ₁ i)]
    [Fintype ι] (b₁ : (i : ι) → Basis (κ₁ i) R (N₁ i)) (b₂ : (i : ι) → Basis (κ₂ i) R (N₂ i))
    {f : M₁ →ₗ[R] M₂} (hf : ∀ i, MapsTo f (N₁ i) (N₂ i)) :
    toMatrix (h₁.collectedBasis b₁) (h₂.collectedBasis b₂) f =
    Matrix.blockDiagonal' fun i ↦ toMatrix (b₁ i) (b₂ i) (f.restrict (hf i)) := by
  /-
    ι : Type u_1
    inst✝⁹ : DecidableEq ι
    R : Type u_4
    M₁ : Type u_5
    M₂ : Type u_6
    inst✝⁸ : CommSemiring R
    inst✝⁷ : AddCommMonoid M₁
    inst✝⁶ : Module R M₁
    N₁ : ι → Submodule R M₁
    h₁ : DirectSum.IsInternal N₁
    inst✝⁵ : AddCommMonoid M₂
    inst✝⁴ : Module R M₂
    N₂ : ι → Submodule R M₂
    h₂ : DirectSum.IsInternal N₂
    κ₁ : ι → Type u_7
    κ₂ : ι → Type u_8
    inst✝³ : (i : ι) → Fintype (κ₁ i)
    inst✝² : ∀ (i : ι), Finite (κ₂ i)
    inst✝¹ : (i : ι) → DecidableEq (κ₁ i)
    inst✝ : Fintype ι
    b₁ : (i : ι) → Basis (κ₁ i) R (Subtype fun x => Membership.mem (N₁ i) x)
    b₂ : (i : ι) → Basis (κ₂ i) R (Subtype fun x => Membership.mem (N₂ i) x)
    f : LinearMap (RingHom.id R) M₁ M₂
    hf : ∀ (i : ι), Set.MapsTo ⇑f ↑(N₁ i) ↑(N₂ i)
    ⊢ Eq ((LinearMap.toMatrix (h₁.collectedBasis b₁) (h₂.collectedBasis b₂)) f) (M …
  -/
  ext ⟨i, _⟩ ⟨j, _⟩
  /-
    case a.mk.mk
    ι : Type u_1
    inst✝⁹ : DecidableEq ι
    R : Type u_4
    M₁ : Type u_5
    M₂ : Type u_6
    inst✝⁸ : CommSemiring R
    inst✝⁷ : AddCommMonoid M₁
    inst✝⁶ : Module R M₁
    N₁ : ι → Submodule R M₁
    h₁ : DirectSum.IsInternal N₁
    inst✝⁵ : AddCommMonoid M₂
    inst✝⁴ : Module R M₂
    N₂ : ι → Submodule R M₂
    h₂ : DirectSum.IsInternal N₂
    κ₁ : ι → Type u_7
    κ₂ : ι → Type u_8
    inst✝³ : (i : ι) → Fintype (κ₁ i)
    inst✝² : ∀ (i : ι), Finite (κ₂ i)
    inst✝¹ : (i : ι) → DecidableEq (κ₁ i)
    inst✝ : Fintype ι
    b₁ : (i : ι) → Basis (κ₁ i) R (Subtype fun x => Membership.mem (N₁ i) x)
    b₂ : (i : ι) → Basis (κ₂ i) R (Subtype fun x => Membership.mem (N₂ i) x)
    f : LinearMap (RingHom.id R) M₁ M₂
    hf : ∀ (i : ι), Set.MapsTo ⇑f ↑(N₁ i) ↑(N₂ i)
    i : ι
    snd✝¹ : κ₂ i
    j : ι
    snd✝ : κ₁ j
    ⊢ Eq ((LinearMap.toMatrix (h₁.collectedBasis b₁) (h₂.collectedBasis b₂)) f ⟨i, …
  -/
  simp only [toMatrix_apply, Matrix.blockDiagonal'_apply]
  /-
    case a.mk.mk
    ι : Type u_1
    inst✝⁹ : DecidableEq ι
    R : Type u_4
    M₁ : Type u_5
    M₂ : Type u_6
    inst✝⁸ : CommSemiring R
    inst✝⁷ : AddCommMonoid M₁
    inst✝⁶ : Module R M₁
    N₁ : ι → Submodule R M₁
    h₁ : DirectSum.IsInternal N₁
    inst✝⁵ : AddCommMonoid M₂
    inst✝⁴ : Module R M₂
    N₂ : ι → Submodule R M₂
    h₂ : DirectSum.IsInternal N₂
    κ₁ : ι → Type u_7
    κ₂ : ι → Type u_8
    inst✝³ : (i : ι) → Fintype (κ₁ i)
    inst✝² : ∀ (i : ι), Finite (κ₂ i)
    inst✝¹ : (i : ι) → DecidableEq (κ₁ i)
    inst✝ : Fintype ι
    b₁ : (i : ι) → Basis (κ₁ i) R (Subtype fun x => Membership.mem (N₁ i) x)
    b₂ : (i : ι) → Basis (κ₂ i) R (Subtype fun x => Membership.mem (N₂ i) x)
    f : LinearMap (RingHom.id R) M₁ M₂
    hf : ∀ (i : ι), Set.MapsTo ⇑f ↑(N₁ i) ↑(N₂ i)
    i : ι
    snd✝¹ : κ₂ i
    j : ι
    snd✝ : κ₁ j
    ⊢ Eq (((h₂.collectedBasis b₂).repr (f ((h₁.collectedBasis b₁) ⟨j, snd✝⟩))) ⟨i, …
  -/
  rcases eq_or_ne i j with rfl | hij
    /-
      case a.mk.mk.inl
      ι : Type u_1
      inst✝⁹ : DecidableEq ι
      R : Type u_4
      M₁ : Type u_5
      M₂ : Type u_6
      inst✝⁸ : CommSemiring R
      inst✝⁷ : AddCommMonoid M₁
      inst✝⁶ : Module R M₁
      N₁ : ι → Submodule R M₁
      h₁ : DirectSum.IsInternal N₁
      inst✝⁵ : AddCommMonoid M₂
      inst✝⁴ : Module R M₂
      N₂ : ι → Submodule R M₂
      h₂ : DirectSum.IsInternal N₂
      κ₁ : ι → Type u_7
      κ₂ : ι → Type u_8
      inst✝³ : (i : ι) → Fintype (κ₁ i)
      inst✝² : ∀ (i : ι), Finite (κ₂ i)
      inst✝¹ : (i : ι) → DecidableEq (κ₁ i)
      inst✝ : Fintype ι
      b₁ : (i : ι) → Basis (κ₁ i) R (Subtype fun x => Membership.mem (N₁ i) x)
      b₂ : (i : ι) → Basis (κ₂ i) R (Subtype fun x => Membership.mem (N₂ i) x)
      f : LinearMap (RingHom.id R) M₁ M₂
      hf : ∀ (i : ι), Set.MapsTo ⇑f ↑(N₁ i) ↑(N₂ i)
      i : ι
      snd✝¹ : κ₂ i
      snd✝ : κ₁ i
      ⊢ Eq (((h₂.collectedBasis b₂).repr (f ((h₁.collectedBasis b₁) ⟨i, snd✝⟩))) ⟨i, …
    -/
  · simp [h₂.collectedBasis_repr_of_mem _ (hf _ (Subtype.mem _)), restrict_apply]
    /-
      🎉 no goals
    -/
    /-
      case a.mk.mk.inr
      ι : Type u_1
      inst✝⁹ : DecidableEq ι
      R : Type u_4
      M₁ : Type u_5
      M₂ : Type u_6
      inst✝⁸ : CommSemiring R
      inst✝⁷ : AddCommMonoid M₁
      inst✝⁶ : Module R M₁
      N₁ : ι → Submodule R M₁
      h₁ : DirectSum.IsInternal N₁
      inst✝⁵ : AddCommMonoid M₂
      inst✝⁴ : Module R M₂
      N₂ : ι → Submodule R M₂
      h₂ : DirectSum.IsInternal N₂
      κ₁ : ι → Type u_7
      κ₂ : ι → Type u_8
      inst✝³ : (i : ι) → Fintype (κ₁ i)
      inst✝² : ∀ (i : ι), Finite (κ₂ i)
      inst✝¹ : (i : ι) → DecidableEq (κ₁ i)
      inst✝ : Fintype ι
      b₁ : (i : ι) → Basis (κ₁ i) R (Subtype fun x => Membership.mem (N₁ i) x)
      b₂ : (i : ι) → Basis (κ₂ i) R (Subtype fun x => Membership.mem (N₂ i) x)
      f : LinearMap (RingHom.id R) M₁ M₂
      hf : ∀ (i : ι), Set.MapsTo ⇑f ↑(N₁ i) ↑(N₂ i)
      i : ι
      snd✝¹ : κ₂ i
      j : ι
      snd✝ : κ₁ j
      hij : Ne i j
      ⊢ Eq (((h₂.collectedBasis b₂).repr (f ((h₁.collectedBasis b₁) ⟨j, snd✝⟩))) ⟨i, …
    -/
  · simp [hij, h₂.collectedBasis_repr_of_mem_ne _ hij.symm (hf _ (Subtype.mem _))]
    /-
      🎉 no goals
    -/


lemma diag_toMatrix_directSum_collectedBasis_eq_zero_of_mapsTo_ne
    {κ : ι → Type*} [∀ i, Fintype (κ i)] [∀ i, DecidableEq (κ i)]
    {s : Finset ι} (h : IsInternal fun i : s ↦ N i)
    (b : (i : s) → Basis (κ i) R (N i)) (σ : ι → ι) (hσ : ∀ i, σ i ≠ i)
    {f : Module.End R M} (hf : ∀ i, MapsTo f (N i) (N <| σ i)) (hN : ∀ i, i ∉ s → N i = ⊥) :
    Matrix.diag (toMatrix (h.collectedBasis b) (h.collectedBasis b) f) = 0 := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    N : ι → Submodule R M
    inst✝² : DecidableEq ι
    κ : ι → Type u_4
    inst✝¹ : (i : ι) → Fintype (κ i)
    inst✝ : (i : ι) → DecidableEq (κ i)
    s : Finset ι
    h : DirectSum.IsInternal fun i => N ↑i
    b : (i : Subtype fun x => Membership.mem s x) → Basis (κ ↑i) R (Subtype fun x  …
    σ : ι → ι
    hσ : ∀ (i : ι), Ne (σ i) i
    f : Module.End R M
    hf : ∀ (i : ι), Set.MapsTo ⇑f ↑(N i) ↑(N (σ i))
    hN : ∀ (i : ι), Not (Membership.mem s i) → Eq (N i) Bot.bot
    ⊢ Eq ((LinearMap.toMatrix (h.collectedBasis b) (h.collectedBasis b)) f).diag 0
  -/
  ext ⟨i, k⟩
  /-
    case h.mk
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    N : ι → Submodule R M
    inst✝² : DecidableEq ι
    κ : ι → Type u_4
    inst✝¹ : (i : ι) → Fintype (κ i)
    inst✝ : (i : ι) → DecidableEq (κ i)
    s : Finset ι
    h : DirectSum.IsInternal fun i => N ↑i
    b : (i : Subtype fun x => Membership.mem s x) → Basis (κ ↑i) R (Subtype fun x  …
    σ : ι → ι
    hσ : ∀ (i : ι), Ne (σ i) i
    f : Module.End R M
    hf : ∀ (i : ι), Set.MapsTo ⇑f ↑(N i) ↑(N (σ i))
    hN : ∀ (i : ι), Not (Membership.mem s i) → Eq (N i) Bot.bot
    i : Subtype fun x => Membership.mem s x
    k : κ ↑i
    ⊢ Eq (((LinearMap.toMatrix (h.collectedBasis b) (h.collectedBasis b)) f).diag  …
  -/
  simp only [Matrix.diag_apply, Pi.zero_apply, toMatrix_apply, IsInternal.collectedBasis_coe]
  /-
    case h.mk
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    N : ι → Submodule R M
    inst✝² : DecidableEq ι
    κ : ι → Type u_4
    inst✝¹ : (i : ι) → Fintype (κ i)
    inst✝ : (i : ι) → DecidableEq (κ i)
    s : Finset ι
    h : DirectSum.IsInternal fun i => N ↑i
    b : (i : Subtype fun x => Membership.mem s x) → Basis (κ ↑i) R (Subtype fun x  …
    σ : ι → ι
    hσ : ∀ (i : ι), Ne (σ i) i
    f : Module.End R M
    hf : ∀ (i : ι), Set.MapsTo ⇑f ↑(N i) ↑(N (σ i))
    hN : ∀ (i : ι), Not (Membership.mem s i) → Eq (N i) Bot.bot
    i : Subtype fun x => Membership.mem s x
    k : κ ↑i
    ⊢ Eq (((h.collectedBasis b).repr (f ↑((b i) k))) ⟨i, k⟩) 0
  -/
  by_cases hi : σ i ∈ s
    /-
      case pos
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      inst✝⁵ : CommRing R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      N : ι → Submodule R M
      inst✝² : DecidableEq ι
      κ : ι → Type u_4
      inst✝¹ : (i : ι) → Fintype (κ i)
      inst✝ : (i : ι) → DecidableEq (κ i)
      s : Finset ι
      h : DirectSum.IsInternal fun i => N ↑i
      b : (i : Subtype fun x => Membership.mem s x) → Basis (κ ↑i) R (Subtype fun x  …
      σ : ι → ι
      hσ : ∀ (i : ι), Ne (σ i) i
      f : Module.End R M
      hf : ∀ (i : ι), Set.MapsTo ⇑f ↑(N i) ↑(N (σ i))
      hN : ∀ (i : ι), Not (Membership.mem s i) → Eq (N i) Bot.bot
      i : Subtype fun x => Membership.mem s x
      k : κ ↑i
      hi : Membership.mem s (σ ↑i)
      ⊢ Eq (((h.collectedBasis b).repr (f ↑((b i) k))) ⟨i, k⟩) 0
    -/
  · let j : s := ⟨σ i, hi⟩
    /-
      case pos
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      inst✝⁵ : CommRing R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      N : ι → Submodule R M
      inst✝² : DecidableEq ι
      κ : ι → Type u_4
      inst✝¹ : (i : ι) → Fintype (κ i)
      inst✝ : (i : ι) → DecidableEq (κ i)
      s : Finset ι
      h : DirectSum.IsInternal fun i => N ↑i
      b : (i : Subtype fun x => Membership.mem s x) → Basis (κ ↑i) R (Subtype fun x  …
      σ : ι → ι
      hσ : ∀ (i : ι), Ne (σ i) i
      f : Module.End R M
      hf : ∀ (i : ι), Set.MapsTo ⇑f ↑(N i) ↑(N (σ i))
      hN : ∀ (i : ι), Not (Membership.mem s i) → Eq (N i) Bot.bot
      i : Subtype fun x => Membership.mem s x
      k : κ ↑i
      hi : Membership.mem s (σ ↑i)
      j : Subtype fun x => Membership.mem s x := ⟨σ ↑i, hi⟩
      ⊢ Eq (((h.collectedBasis b).repr (f ↑((b i) k))) ⟨i, k⟩) 0
    -/
    replace hσ : j ≠ i := fun hij ↦ hσ i <| Subtype.ext_iff.mp hij
    /-
      case pos
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      inst✝⁵ : CommRing R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      N : ι → Submodule R M
      inst✝² : DecidableEq ι
      κ : ι → Type u_4
      inst✝¹ : (i : ι) → Fintype (κ i)
      inst✝ : (i : ι) → DecidableEq (κ i)
      s : Finset ι
      h : DirectSum.IsInternal fun i => N ↑i
      b : (i : Subtype fun x => Membership.mem s x) → Basis (κ ↑i) R (Subtype fun x  …
      σ : ι → ι
      f : Module.End R M
      hf : ∀ (i : ι), Set.MapsTo ⇑f ↑(N i) ↑(N (σ i))
      hN : ∀ (i : ι), Not (Membership.mem s i) → Eq (N i) Bot.bot
      i : Subtype fun x => Membership.mem s x
      k : κ ↑i
      hi : Membership.mem s (σ ↑i)
      j : Subtype fun x => Membership.mem s x := ⟨σ ↑i, hi⟩
      hσ : Ne j i
      ⊢ Eq (((h.collectedBasis b).repr (f ↑((b i) k))) ⟨i, k⟩) 0
    -/
    exact h.collectedBasis_repr_of_mem_ne b hσ <| hf _ <| Subtype.mem (b i k)
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      inst✝⁵ : CommRing R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      N : ι → Submodule R M
      inst✝² : DecidableEq ι
      κ : ι → Type u_4
      inst✝¹ : (i : ι) → Fintype (κ i)
      inst✝ : (i : ι) → DecidableEq (κ i)
      s : Finset ι
      h : DirectSum.IsInternal fun i => N ↑i
      b : (i : Subtype fun x => Membership.mem s x) → Basis (κ ↑i) R (Subtype fun x  …
      σ : ι → ι
      hσ : ∀ (i : ι), Ne (σ i) i
      f : Module.End R M
      hf : ∀ (i : ι), Set.MapsTo ⇑f ↑(N i) ↑(N (σ i))
      hN : ∀ (i : ι), Not (Membership.mem s i) → Eq (N i) Bot.bot
      i : Subtype fun x => Membership.mem s x
      k : κ ↑i
      hi : Not (Membership.mem s (σ ↑i))
      ⊢ Eq (((h.collectedBasis b).repr (f ↑((b i) k))) ⟨i, k⟩) 0
    -/
  · suffices f (b i k) = 0 by simp [this]
    /-
      case neg
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      inst✝⁵ : CommRing R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      N : ι → Submodule R M
      inst✝² : DecidableEq ι
      κ : ι → Type u_4
      inst✝¹ : (i : ι) → Fintype (κ i)
      inst✝ : (i : ι) → DecidableEq (κ i)
      s : Finset ι
      h : DirectSum.IsInternal fun i => N ↑i
      b : (i : Subtype fun x => Membership.mem s x) → Basis (κ ↑i) R (Subtype fun x  …
      σ : ι → ι
      hσ : ∀ (i : ι), Ne (σ i) i
      f : Module.End R M
      hf : ∀ (i : ι), Set.MapsTo ⇑f ↑(N i) ↑(N (σ i))
      hN : ∀ (i : ι), Not (Membership.mem s i) → Eq (N i) Bot.bot
      i : Subtype fun x => Membership.mem s x
      k : κ ↑i
      hi : Not (Membership.mem s (σ ↑i))
      ⊢ Eq (f ↑((b i) k)) 0
    -/
    simpa [hN _ hi] using hf i <| Subtype.mem (b i k)
    /-
      🎉 no goals
    -/


/-- The trace of an endomorphism of a direct sum is the sum of the traces on each component.

See also `LinearMap.trace_restrict_eq_sum_trace_restrict`. -/
lemma trace_eq_sum_trace_restrict (h : IsInternal N) [Fintype ι]
    {f : M →ₗ[R] M} (hf : ∀ i, MapsTo f (N i) (N i)) :
    trace R M f = ∑ i, trace R (N i) (f.restrict (hf i)) := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    N : ι → Submodule R M
    inst✝³ : DecidableEq ι
    inst✝² : ∀ (i : ι), Module.Finite R (Subtype fun x => Membership.mem (N i) x)
    inst✝¹ : ∀ (i : ι), Module.Free R (Subtype fun x => Membership.mem (N i) x)
    h : DirectSum.IsInternal N
    inst✝ : Fintype ι
    f : LinearMap (RingHom.id R) M M
    hf : ∀ (i : ι), Set.MapsTo ⇑f ↑(N i) ↑(N i)
    ⊢ Eq ((LinearMap.trace R M) f) (Finset.univ.sum fun i => (LinearMap.trace R (S …
  -/
  let b : (i : ι) → Basis _ R (N i) := fun i ↦ Module.Free.chooseBasis R (N i)
  simp_rw [trace_eq_matrix_trace R (h.collectedBasis b),
    toMatrix_directSum_collectedBasis_eq_blockDiagonal' h h b b hf, Matrix.trace_blockDiagonal',
    ← trace_eq_matrix_trace]


lemma trace_eq_sum_trace_restrict' (h : IsInternal N) (hN : {i | N i ≠ ⊥}.Finite)
    {f : M →ₗ[R] M} (hf : ∀ i, MapsTo f (N i) (N i)) :
    trace R M f = ∑ i ∈ hN.toFinset, trace R (N i) (f.restrict (hf i)) := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    N : ι → Submodule R M
    inst✝² : DecidableEq ι
    inst✝¹ : ∀ (i : ι), Module.Finite R (Subtype fun x => Membership.mem (N i) x)
    inst✝ : ∀ (i : ι), Module.Free R (Subtype fun x => Membership.mem (N i) x)
    h : DirectSum.IsInternal N
    hN : (setOf fun i => Ne (N i) Bot.bot).Finite
    f : LinearMap (RingHom.id R) M M
    hf : ∀ (i : ι), Set.MapsTo ⇑f ↑(N i) ↑(N i)
    ⊢ Eq ((LinearMap.trace R M) f) (hN.toFinset.sum fun i => (LinearMap.trace R (S …
  -/
  let _ : Fintype {i // N i ≠ ⊥} := hN.fintype
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    N : ι → Submodule R M
    inst✝² : DecidableEq ι
    inst✝¹ : ∀ (i : ι), Module.Finite R (Subtype fun x => Membership.mem (N i) x)
    inst✝ : ∀ (i : ι), Module.Free R (Subtype fun x => Membership.mem (N i) x)
    h : DirectSum.IsInternal N
    hN : (setOf fun i => Ne (N i) Bot.bot).Finite
    f : LinearMap (RingHom.id R) M M
    hf : ∀ (i : ι), Set.MapsTo ⇑f ↑(N i) ↑(N i)
    x✝ : Fintype (Subtype fun i => Ne (N i) Bot.bot) := hN.fintype
    ⊢ Eq ((LinearMap.trace R M) f) (hN.toFinset.sum fun i => (LinearMap.trace R (S …
  -/
  let _ : Fintype {i | N i ≠ ⊥} := hN.fintype
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    N : ι → Submodule R M
    inst✝² : DecidableEq ι
    inst✝¹ : ∀ (i : ι), Module.Finite R (Subtype fun x => Membership.mem (N i) x)
    inst✝ : ∀ (i : ι), Module.Free R (Subtype fun x => Membership.mem (N i) x)
    h : DirectSum.IsInternal N
    hN : (setOf fun i => Ne (N i) Bot.bot).Finite
    f : LinearMap (RingHom.id R) M M
    hf : ∀ (i : ι), Set.MapsTo ⇑f ↑(N i) ↑(N i)
    x✝¹ : Fintype (Subtype fun i => Ne (N i) Bot.bot) := hN.fintype
    x✝ : Fintype ↑(setOf fun i => Ne (N i) Bot.bot) := hN.fintype
    ⊢ Eq ((LinearMap.trace R M) f) (hN.toFinset.sum fun i => (LinearMap.trace R (S …
  -/
  rw [← Finset.sum_coe_sort, trace_eq_sum_trace_restrict (isInternal_ne_bot_iff.mpr h) _]
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    N : ι → Submodule R M
    inst✝² : DecidableEq ι
    inst✝¹ : ∀ (i : ι), Module.Finite R (Subtype fun x => Membership.mem (N i) x)
    inst✝ : ∀ (i : ι), Module.Free R (Subtype fun x => Membership.mem (N i) x)
    h : DirectSum.IsInternal N
    hN : (setOf fun i => Ne (N i) Bot.bot).Finite
    f : LinearMap (RingHom.id R) M M
    hf : ∀ (i : ι), Set.MapsTo ⇑f ↑(N i) ↑(N i)
    x✝¹ : Fintype (Subtype fun i => Ne (N i) Bot.bot) := hN.fintype
    x✝ : Fintype ↑(setOf fun i => Ne (N i) Bot.bot) := hN.fintype
    ⊢ Eq (Finset.univ.sum fun i => (LinearMap.trace R (Subtype fun x => Membership …
  -/
  exact Fintype.sum_equiv hN.subtypeEquivToFinset _ _ (fun i ↦ rfl)
  /-
    🎉 no goals
  -/


lemma trace_eq_zero_of_mapsTo_ne (h : IsInternal N) [IsNoetherian R M]
    (σ : ι → ι) (hσ : ∀ i, σ i ≠ i) {f : Module.End R M}
    (hf : ∀ i, MapsTo f (N i) (N <| σ i)) :
    trace R M f = 0 := by
  have hN : {i | N i ≠ ⊥}.Finite := WellFoundedGT.finite_ne_bot_of_iSupIndep
    h.submodule_iSupIndep
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    N : ι → Submodule R M
    inst✝³ : DecidableEq ι
    inst✝² : ∀ (i : ι), Module.Finite R (Subtype fun x => Membership.mem (N i) x)
    inst✝¹ : ∀ (i : ι), Module.Free R (Subtype fun x => Membership.mem (N i) x)
    h : DirectSum.IsInternal N
    inst✝ : IsNoetherian R M
    σ : ι → ι
    hσ : ∀ (i : ι), Ne (σ i) i
    f : Module.End R M
    hf : ∀ (i : ι), Set.MapsTo ⇑f ↑(N i) ↑(N (σ i))
    hN : (setOf fun i => Ne (N i) Bot.bot).Finite
    ⊢ Eq ((LinearMap.trace R M) f) 0
  -/
  let s := hN.toFinset
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    N : ι → Submodule R M
    inst✝³ : DecidableEq ι
    inst✝² : ∀ (i : ι), Module.Finite R (Subtype fun x => Membership.mem (N i) x)
    inst✝¹ : ∀ (i : ι), Module.Free R (Subtype fun x => Membership.mem (N i) x)
    h : DirectSum.IsInternal N
    inst✝ : IsNoetherian R M
    σ : ι → ι
    hσ : ∀ (i : ι), Ne (σ i) i
    f : Module.End R M
    hf : ∀ (i : ι), Set.MapsTo ⇑f ↑(N i) ↑(N (σ i))
    hN : (setOf fun i => Ne (N i) Bot.bot).Finite
    s : Finset ι := hN.toFinset
    ⊢ Eq ((LinearMap.trace R M) f) 0
  -/
  let κ := fun i ↦ Module.Free.ChooseBasisIndex R (N i)
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    N : ι → Submodule R M
    inst✝³ : DecidableEq ι
    inst✝² : ∀ (i : ι), Module.Finite R (Subtype fun x => Membership.mem (N i) x)
    inst✝¹ : ∀ (i : ι), Module.Free R (Subtype fun x => Membership.mem (N i) x)
    h : DirectSum.IsInternal N
    inst✝ : IsNoetherian R M
    σ : ι → ι
    hσ : ∀ (i : ι), Ne (σ i) i
    f : Module.End R M
    hf : ∀ (i : ι), Set.MapsTo ⇑f ↑(N i) ↑(N (σ i))
    hN : (setOf fun i => Ne (N i) Bot.bot).Finite
    s : Finset ι := hN.toFinset
    κ : ι → Type u_3 := fun i => Module.Free.ChooseBasisIndex R (Subtype fun x =>  …
    ⊢ Eq ((LinearMap.trace R M) f) 0
  -/
  let b : (i : s) → Basis (κ i) R (N i) := fun i ↦ Module.Free.chooseBasis R (N i)
  replace h : IsInternal fun i : s ↦ N i := by
    convert DirectSum.isInternal_ne_bot_iff.mpr h <;> simp [s]
  simp_rw [trace_eq_matrix_trace R (h.collectedBasis b), Matrix.trace,
    diag_toMatrix_directSum_collectedBasis_eq_zero_of_mapsTo_ne h b σ hσ hf (by simp [s]),
    Pi.zero_apply, Finset.sum_const_zero]


/-- If `f` and `g` are commuting endomorphisms of a finite, free `R`-module `M`, such that `f`
is triangularizable, then to prove that the trace of `g ∘ f` vanishes, it is sufficient to prove
that the trace of `g` vanishes on each generalized eigenspace of `f`. -/
lemma trace_comp_eq_zero_of_commute_of_trace_restrict_eq_zero
    [IsDomain R] [IsPrincipalIdealRing R] [Module.Free R M] [Module.Finite R M]
    {f g : Module.End R M}
    (h_comm : Commute f g)
    (hf : ⨆ μ, f.maxGenEigenspace μ = ⊤)
    (hg : ∀ μ, trace R _ (g.restrict (f.mapsTo_maxGenEigenspace_of_comm h_comm μ)) = 0) :
    trace R _ (g ∘ₗ f) = 0 := by
  have hfg : ∀ μ,
      MapsTo (g ∘ₗ f) ↑(f.maxGenEigenspace μ) ↑(f.maxGenEigenspace μ) :=
    fun μ ↦ (f.mapsTo_maxGenEigenspace_of_comm h_comm μ).comp
      (f.mapsTo_maxGenEigenspace_of_comm rfl μ)
  suffices ∀ μ, trace R _ ((g ∘ₗ f).restrict (hfg μ)) = 0 by
    classical
    have hds := DirectSum.isInternal_submodule_of_iSupIndep_of_iSup_eq_top
      f.independent_maxGenEigenspace hf
    have h_fin : {μ | f.maxGenEigenspace μ ≠ ⊥}.Finite :=
      WellFoundedGT.finite_ne_bot_of_iSupIndep f.independent_maxGenEigenspace
    simp [trace_eq_sum_trace_restrict' hds h_fin hfg, this]
  /-
    R : Type u_2
    M : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : IsDomain R
    inst✝² : IsPrincipalIdealRing R
    inst✝¹ : Module.Free R M
    inst✝ : Module.Finite R M
    f g : Module.End R M
    h_comm : Commute f g
    hf : Eq (iSup fun μ => f.maxGenEigenspace μ) Top.top
    hg : ∀ (μ : R), Eq ((LinearMap.trace R (Subtype fun x => Membership.mem (f.max …
    hfg : ∀ (μ : R), Set.MapsTo ⇑(LinearMap.comp g f) ↑(f.maxGenEigenspace μ) ↑(f. …
    ⊢ ∀ (μ : R), Eq ((LinearMap.trace R (Subtype fun x => Membership.mem (f.maxGen …
  -/
  intro μ
  replace h_comm : Commute (g.restrict (f.mapsTo_maxGenEigenspace_of_comm h_comm μ))
      (f.restrict (f.mapsTo_maxGenEigenspace_of_comm rfl μ)) :=
    restrict_commute h_comm.symm _ _
  rw [restrict_comp, trace_comp_eq_mul_of_commute_of_isNilpotent μ h_comm
    (f.isNilpotent_restrict_maxGenEigenspace_sub_algebraMap μ), hg, mul_zero]


lemma mapsTo_biSup_of_mapsTo {ι : Type*} {N : ι → Submodule R M}
    (s : Set ι) {f : Module.End R M} (hf : ∀ i, MapsTo f (N i) (N i)) :
    MapsTo f ↑(⨆ i ∈ s, N i) ↑(⨆ i ∈ s, N i) := by
  /-
    R : Type u_2
    M : Type u_3
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ι : Type u_4
    N : ι → Submodule R M
    s : Set ι
    f : Module.End R M
    hf : ∀ (i : ι), Set.MapsTo ⇑f ↑(N i) ↑(N i)
    ⊢ Set.MapsTo ⇑f ↑(iSup fun i => iSup fun h => N i) ↑(iSup fun i => iSup fun h  …
  -/
  replace hf : ∀ i, (N i).map f ≤ N i := fun i ↦ Submodule.map_le_iff_le_comap.mpr (hf i)
  /-
    R : Type u_2
    M : Type u_3
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ι : Type u_4
    N : ι → Submodule R M
    s : Set ι
    f : Module.End R M
    hf : ∀ (i : ι), LE.le (Submodule.map f (N i)) (N i)
    ⊢ Set.MapsTo ⇑f ↑(iSup fun i => iSup fun h => N i) ↑(iSup fun i => iSup fun h  …
  -/
  suffices (⨆ i ∈ s, N i).map f ≤ ⨆ i ∈ s, N i from Submodule.map_le_iff_le_comap.mp this
  /-
    R : Type u_2
    M : Type u_3
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ι : Type u_4
    N : ι → Submodule R M
    s : Set ι
    f : Module.End R M
    hf : ∀ (i : ι), LE.le (Submodule.map f (N i)) (N i)
    ⊢ LE.le (Submodule.map f (iSup fun i => iSup fun h => N i)) (iSup fun i => iSu …
  -/
  simpa only [Submodule.map_iSup] using iSup₂_mono <| fun i _ ↦ hf i
  /-
    🎉 no goals
  -/


/-- The trace of an endomorphism of a direct sum is the sum of the traces on each component.

Note that it is important the statement gives the user definitional control over `p` since the
_type_ of the term `trace R p (f.restrict hp')` depends on `p`. -/
lemma trace_eq_sum_trace_restrict_of_eq_biSup
    [∀ i, Module.Finite R (N i)] [∀ i, Module.Free R (N i)]
    (s : Finset ι) (h : iSupIndep <| fun i : s ↦ N i)
    {f : Module.End R M} (hf : ∀ i, MapsTo f (N i) (N i))
    (p : Submodule R M) (hp : p = ⨆ i ∈ s, N i)
    (hp' : MapsTo f p p := hp ▸ mapsTo_biSup_of_mapsTo (s : Set ι) hf) :
    trace R p (f.restrict hp') = ∑ i ∈ s, trace R (N i) (f.restrict (hf i)) := by
  classical
  let N' : s → Submodule R p := fun i ↦ (N i).comap p.subtype
  replace h : IsInternal N' := hp ▸ isInternal_biSup_submodule_of_iSupIndep (s : Set ι) h
  have hf' : ∀ i, MapsTo (restrict f hp') (N' i) (N' i) := fun i x hx' ↦ by simpa using hf i hx'
  let e : (i : s) → N' i ≃ₗ[R] N i := fun ⟨i, hi⟩ ↦ (N i).comapSubtypeEquivOfLe (hp ▸ le_biSup N hi)
  have _i1 : ∀ i, Module.Finite R (N' i) := fun i ↦ Module.Finite.equiv (e i).symm
  have _i2 : ∀ i, Module.Free R (N' i) := fun i ↦ Module.Free.of_equiv (e i).symm
  rw [trace_eq_sum_trace_restrict h hf', ← s.sum_coe_sort]
  have : ∀ i : s, f.restrict (hf i) = (e i).conj ((f.restrict hp').restrict (hf' i)) := fun _ ↦ rfl
  simp [this]


