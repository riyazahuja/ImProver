/-- Interpret a `Finsupp` as a homogeneous `DFinsupp`. -/
def Finsupp.toDFinsupp [Zero M] (f : ι →₀ M) : Π₀ _ : ι, M where
  toFun := f
  support' :=
    Trunc.mk
      ⟨f.support.1, fun i => (Classical.em (f i = 0)).symm.imp_left Finsupp.mem_support_iff.mpr⟩


@[simp]
theorem Finsupp.toDFinsupp_coe [Zero M] (f : ι →₀ M) : ⇑f.toDFinsupp = f :=
  rfl


@[simp]
theorem Finsupp.toDFinsupp_single (i : ι) (m : M) :
    (Finsupp.single i m).toDFinsupp = DFinsupp.single i m := by
  /-
    ι : Type u_1
    M : Type u_3
    inst✝¹ : DecidableEq ι
    inst✝ : Zero M
    i : ι
    m : M
    ⊢ Eq (Finsupp.single i m).toDFinsupp (DFinsupp.single i m)
  -/
  ext
  /-
    case h
    ι : Type u_1
    M : Type u_3
    inst✝¹ : DecidableEq ι
    inst✝ : Zero M
    i : ι
    m : M
    i✝ : ι
    ⊢ Eq ((Finsupp.single i m).toDFinsupp i✝) ((DFinsupp.single i m) i✝)
  -/
  simp [Finsupp.single_apply, DFinsupp.single_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem toDFinsupp_support (f : ι →₀ M) : f.toDFinsupp.support = f.support := by
  /-
    ι : Type u_1
    M : Type u_3
    inst✝² : DecidableEq ι
    inst✝¹ : Zero M
    inst✝ : (m : M) → Decidable (Ne m 0)
    f : Finsupp ι M
    ⊢ Eq f.toDFinsupp.support f.support
  -/
  ext
  /-
    case h
    ι : Type u_1
    M : Type u_3
    inst✝² : DecidableEq ι
    inst✝¹ : Zero M
    inst✝ : (m : M) → Decidable (Ne m 0)
    f : Finsupp ι M
    a✝ : ι
    ⊢ Iff (Membership.mem f.toDFinsupp.support a✝) (Membership.mem f.support a✝)
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Interpret a homogeneous `DFinsupp` as a `Finsupp`.

Note that the elaborator has a lot of trouble with this definition - it is often necessary to
write `(DFinsupp.toFinsupp f : ι →₀ M)` instead of `f.toFinsupp`, as for some unknown reason
using dot notation or omitting the type ascription prevents the type being resolved correctly. -/
def DFinsupp.toFinsupp (f : Π₀ _ : ι, M) : ι →₀ M :=
                             /-
                               ι : Type u_1
                               R : Type u_2
                               M : Type u_3
                               inst✝² : DecidableEq ι
                               inst✝¹ : Zero M
                               inst✝ : (m : M) → Decidable (Ne m 0)
                               f : DFinsupp fun x => M
                               i : ι
                               ⊢ Iff (Membership.mem f.support i) (Ne (f i) 0)
                             -/
  ⟨f.support, f, fun i => by simp only [DFinsupp.mem_support_iff]⟩
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem DFinsupp.toFinsupp_coe (f : Π₀ _ : ι, M) : ⇑f.toFinsupp = f :=
  rfl


@[simp]
theorem DFinsupp.toFinsupp_support (f : Π₀ _ : ι, M) : f.toFinsupp.support = f.support := by
  /-
    ι : Type u_1
    M : Type u_3
    inst✝² : DecidableEq ι
    inst✝¹ : Zero M
    inst✝ : (m : M) → Decidable (Ne m 0)
    f : DFinsupp fun x => M
    ⊢ Eq f.toFinsupp.support f.support
  -/
  ext
  /-
    case h
    ι : Type u_1
    M : Type u_3
    inst✝² : DecidableEq ι
    inst✝¹ : Zero M
    inst✝ : (m : M) → Decidable (Ne m 0)
    f : DFinsupp fun x => M
    a✝ : ι
    ⊢ Iff (Membership.mem f.toFinsupp.support a✝) (Membership.mem f.support a✝)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem DFinsupp.toFinsupp_single (i : ι) (m : M) :
    (DFinsupp.single i m : Π₀ _ : ι, M).toFinsupp = Finsupp.single i m := by
  /-
    ι : Type u_1
    M : Type u_3
    inst✝² : DecidableEq ι
    inst✝¹ : Zero M
    inst✝ : (m : M) → Decidable (Ne m 0)
    i : ι
    m : M
    ⊢ Eq (DFinsupp.single i m).toFinsupp (Finsupp.single i m)
  -/
  ext
  /-
    case h
    ι : Type u_1
    M : Type u_3
    inst✝² : DecidableEq ι
    inst✝¹ : Zero M
    inst✝ : (m : M) → Decidable (Ne m 0)
    i : ι
    m : M
    a✝ : ι
    ⊢ Eq ((DFinsupp.single i m).toFinsupp a✝) ((Finsupp.single i m) a✝)
  -/
  simp [Finsupp.single_apply, DFinsupp.single_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem Finsupp.toDFinsupp_toFinsupp (f : ι →₀ M) : f.toDFinsupp.toFinsupp = f :=
  DFunLike.coe_injective rfl


@[simp]
theorem DFinsupp.toFinsupp_toDFinsupp (f : Π₀ _ : ι, M) : f.toFinsupp.toDFinsupp = f :=
  DFunLike.coe_injective rfl


@[simp]
theorem toDFinsupp_zero [Zero M] : (0 : ι →₀ M).toDFinsupp = 0 :=
  DFunLike.coe_injective rfl


@[simp]
theorem toDFinsupp_add [AddZeroClass M] (f g : ι →₀ M) :
    (f + g).toDFinsupp = f.toDFinsupp + g.toDFinsupp :=
  DFunLike.coe_injective rfl


@[simp]
theorem toDFinsupp_neg [AddGroup M] (f : ι →₀ M) : (-f).toDFinsupp = -f.toDFinsupp :=
  DFunLike.coe_injective rfl


@[simp]
theorem toDFinsupp_sub [AddGroup M] (f g : ι →₀ M) :
    (f - g).toDFinsupp = f.toDFinsupp - g.toDFinsupp :=
  DFunLike.coe_injective rfl


@[simp]
theorem toDFinsupp_smul [Monoid R] [AddMonoid M] [DistribMulAction R M] (r : R) (f : ι →₀ M) :
    (r • f).toDFinsupp = r • f.toDFinsupp :=
  DFunLike.coe_injective rfl


@[simp]
theorem toFinsupp_zero [Zero M] [∀ m : M, Decidable (m ≠ 0)] : toFinsupp 0 = (0 : ι →₀ M) :=
  DFunLike.coe_injective rfl


@[simp]
theorem toFinsupp_add [AddZeroClass M] [∀ m : M, Decidable (m ≠ 0)] (f g : Π₀ _ : ι, M) :
    (toFinsupp (f + g) : ι →₀ M) = toFinsupp f + toFinsupp g :=
  DFunLike.coe_injective <| DFinsupp.coe_add _ _


@[simp]
theorem toFinsupp_neg [AddGroup M] [∀ m : M, Decidable (m ≠ 0)] (f : Π₀ _ : ι, M) :
    (toFinsupp (-f) : ι →₀ M) = -toFinsupp f :=
  DFunLike.coe_injective <| DFinsupp.coe_neg _


@[simp]
theorem toFinsupp_sub [AddGroup M] [∀ m : M, Decidable (m ≠ 0)] (f g : Π₀ _ : ι, M) :
    (toFinsupp (f - g) : ι →₀ M) = toFinsupp f - toFinsupp g :=
  DFunLike.coe_injective <| DFinsupp.coe_sub _ _


@[simp]
theorem toFinsupp_smul [Monoid R] [AddMonoid M] [DistribMulAction R M] [∀ m : M, Decidable (m ≠ 0)]
    (r : R) (f : Π₀ _ : ι, M) : (toFinsupp (r • f) : ι →₀ M) = r • toFinsupp f :=
  DFunLike.coe_injective <| DFinsupp.coe_smul _ _


/-- `Finsupp.toDFinsupp` and `DFinsupp.toFinsupp` together form an equiv. -/
@[simps (config := .asFn)]
def finsuppEquivDFinsupp [DecidableEq ι] [Zero M] [∀ m : M, Decidable (m ≠ 0)] :
    (ι →₀ M) ≃ Π₀ _ : ι, M where
  toFun := Finsupp.toDFinsupp
  invFun := DFinsupp.toFinsupp
  left_inv := Finsupp.toDFinsupp_toFinsupp
  right_inv := DFinsupp.toFinsupp_toDFinsupp


/-- The additive version of `finsupp.toFinsupp`. Note that this is `noncomputable` because
`Finsupp.add` is noncomputable. -/
@[simps (config := .asFn)]
def finsuppAddEquivDFinsupp [DecidableEq ι] [AddZeroClass M] [∀ m : M, Decidable (m ≠ 0)] :
    (ι →₀ M) ≃+ Π₀ _ : ι, M :=
  { finsuppEquivDFinsupp with
    toFun := Finsupp.toDFinsupp
    invFun := DFinsupp.toFinsupp
    map_add' := Finsupp.toDFinsupp_add }


/-- The additive version of `Finsupp.toFinsupp`. Note that this is `noncomputable` because
`Finsupp.add` is noncomputable. -/
-- Porting note: `simps` generated lemmas that did not pass `simpNF` lints, manually added below
--@[simps? (config := .asFn)]
def finsuppLequivDFinsupp [DecidableEq ι] [Semiring R] [AddCommMonoid M]
    [∀ m : M, Decidable (m ≠ 0)] [Module R M] : (ι →₀ M) ≃ₗ[R] Π₀ _ : ι, M :=
  { finsuppEquivDFinsupp with
    toFun := Finsupp.toDFinsupp
    invFun := DFinsupp.toFinsupp
    map_smul' := Finsupp.toDFinsupp_smul
    map_add' := Finsupp.toDFinsupp_add }

-- Porting note: `simps` generated as `↑(finsuppLequivDFinsupp R).toLinearMap = Finsupp.toDFinsupp`

@[simp]
theorem finsuppLequivDFinsupp_apply_apply [DecidableEq ι] [Semiring R] [AddCommMonoid M]
    [∀ m : M, Decidable (m ≠ 0)] [Module R M] :
    (↑(finsuppLequivDFinsupp (M := M) R) : (ι →₀ M) → _) = Finsupp.toDFinsupp := rfl


@[simp]
theorem finsuppLequivDFinsupp_symm_apply [DecidableEq ι] [Semiring R] [AddCommMonoid M]
    [∀ m : M, Decidable (m ≠ 0)] [Module R M] :
    ↑(LinearEquiv.symm (finsuppLequivDFinsupp (ι := ι) (M := M) R)) = DFinsupp.toFinsupp :=
  rfl

-- Porting note: moved noncomputable declaration into section begin

/-- `Finsupp.split` is an equivalence between `(Σ i, η i) →₀ N` and `Π₀ i, (η i →₀ N)`. -/
def sigmaFinsuppEquivDFinsupp [Zero N] : ((Σi, η i) →₀ N) ≃ Π₀ i, η i →₀ N where
  toFun f := ⟨split f, Trunc.mk ⟨(splitSupport f : Finset ι).val, fun i => by
          /-
            ι : Type u_1
            R : Type u_2
            M : Type u_3
            η : ι → Type u_4
            N : Type u_5
            inst✝¹ : Semiring R
            inst✝ : Zero N
            f : Finsupp (Sigma fun i => η i) N
            i : ι
            ⊢ Or (Membership.mem f.splitSupport.val i) (Eq (f.split i) 0)
          -/
          rw [← Finset.mem_def, mem_splitSupport_iff_nonzero]
          /-
            ι : Type u_1
            R : Type u_2
            M : Type u_3
            η : ι → Type u_4
            N : Type u_5
            inst✝¹ : Semiring R
            inst✝ : Zero N
            f : Finsupp (Sigma fun i => η i) N
            i : ι
            ⊢ Or (Ne (f.split i) 0) (Eq (f.split i) 0)
          -/
          exact (em _).symm⟩⟩
          /-
            🎉 no goals
          -/
  invFun f := by
    /-
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      η : ι → Type u_4
      N : Type u_5
      inst✝¹ : Semiring R
      inst✝ : Zero N
      f : DFinsupp fun i => Finsupp (η i) N
      ⊢ Finsupp (Sigma fun i => η i) N
    -/
    haveI := Classical.decEq ι
    /-
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      η : ι → Type u_4
      N : Type u_5
      inst✝¹ : Semiring R
      inst✝ : Zero N
      f : DFinsupp fun i => Finsupp (η i) N
      this : DecidableEq ι
      ⊢ Finsupp (Sigma fun i => η i) N
    -/
    haveI := fun i => Classical.decEq (η i →₀ N)
    refine
      onFinset (Finset.sigma f.support fun j => (f j).support) (fun ji => f ji.1 ji.2) fun g hg =>
        Finset.mem_sigma.mpr ⟨?_, mem_support_iff.mpr hg⟩
    /-
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      η : ι → Type u_4
      N : Type u_5
      inst✝¹ : Semiring R
      inst✝ : Zero N
      f : DFinsupp fun i => Finsupp (η i) N
      this✝ : DecidableEq ι
      this : (i : ι) → DecidableEq (Finsupp (η i) N)
      g : Sigma fun i => η i
      hg : Ne ((fun ji => (f ji.fst) ji.snd) g) 0
      ⊢ Membership.mem f.support g.fst
    -/
    simp only [Ne, DFinsupp.mem_support_toFun]
    /-
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      η : ι → Type u_4
      N : Type u_5
      inst✝¹ : Semiring R
      inst✝ : Zero N
      f : DFinsupp fun i => Finsupp (η i) N
      this✝ : DecidableEq ι
      this : (i : ι) → DecidableEq (Finsupp (η i) N)
      g : Sigma fun i => η i
      hg : Ne ((fun ji => (f ji.fst) ji.snd) g) 0
      ⊢ Not (Eq (f g.fst) 0)
    -/
    intro h
    /-
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      η : ι → Type u_4
      N : Type u_5
      inst✝¹ : Semiring R
      inst✝ : Zero N
      f : DFinsupp fun i => Finsupp (η i) N
      this✝ : DecidableEq ι
      this : (i : ι) → DecidableEq (Finsupp (η i) N)
      g : Sigma fun i => η i
      hg : Ne ((fun ji => (f ji.fst) ji.snd) g) 0
      h : Eq (f g.fst) 0
      ⊢ False
    -/
    dsimp at hg
    /-
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      η : ι → Type u_4
      N : Type u_5
      inst✝¹ : Semiring R
      inst✝ : Zero N
      f : DFinsupp fun i => Finsupp (η i) N
      this✝ : DecidableEq ι
      this : (i : ι) → DecidableEq (Finsupp (η i) N)
      g : Sigma fun i => η i
      hg : Not (Eq ((f g.fst) g.snd) 0)
      h : Eq (f g.fst) 0
      ⊢ False
    -/
    rw [h] at hg
    /-
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      η : ι → Type u_4
      N : Type u_5
      inst✝¹ : Semiring R
      inst✝ : Zero N
      f : DFinsupp fun i => Finsupp (η i) N
      this✝ : DecidableEq ι
      this : (i : ι) → DecidableEq (Finsupp (η i) N)
      g : Sigma fun i => η i
      hg : Not (Eq (0 g.snd) 0)
      h : Eq (f g.fst) 0
      ⊢ False
    -/
    simp only [coe_zero, Pi.zero_apply, not_true] at hg
    /-
      🎉 no goals
    -/
                   /-
                     ι : Type u_1
                     R : Type u_2
                     M : Type u_3
                     η : ι → Type u_4
                     N : Type u_5
                     inst✝¹ : Semiring R
                     inst✝ : Zero N
                     f : Finsupp (Sigma fun i => η i) N
                     ⊢ Eq ((fun f => Finsupp.onFinset (f.support.sigma fun j => (f j).support) (fun …
                   -/
  left_inv f := by ext; simp [split]
                        /-
                          🎉 no goals
                        -/
                    /-
                      ι : Type u_1
                      R : Type u_2
                      M : Type u_3
                      η : ι → Type u_4
                      N : Type u_5
                      inst✝¹ : Semiring R
                      inst✝ : Zero N
                      f : DFinsupp fun i => Finsupp (η i) N
                      ⊢ Eq ((fun f => { toFun := f.split, support' := Trunc.mk ⟨f.splitSupport.val,  …
                    -/
  right_inv f := by ext; simp [split]
                         /-
                           🎉 no goals
                         -/


@[simp]
theorem sigmaFinsuppEquivDFinsupp_apply [Zero N] (f : (Σi, η i) →₀ N) :
    (sigmaFinsuppEquivDFinsupp f : ∀ i, η i →₀ N) = Finsupp.split f :=
  rfl


@[simp]
theorem sigmaFinsuppEquivDFinsupp_symm_apply [Zero N] (f : Π₀ i, η i →₀ N) (s : Σi, η i) :
    (sigmaFinsuppEquivDFinsupp.symm f : (Σi, η i) →₀ N) s = f s.1 s.2 :=
  rfl


@[simp]
theorem sigmaFinsuppEquivDFinsupp_support [DecidableEq ι] [Zero N]
    [∀ (i : ι) (x : η i →₀ N), Decidable (x ≠ 0)] (f : (Σi, η i) →₀ N) :
    (sigmaFinsuppEquivDFinsupp f).support = Finsupp.splitSupport f := by
  /-
    ι : Type u_1
    η : ι → Type u_4
    N : Type u_5
    inst✝² : DecidableEq ι
    inst✝¹ : Zero N
    inst✝ : (i : ι) → (x : Finsupp (η i) N) → Decidable (Ne x 0)
    f : Finsupp (Sigma fun i => η i) N
    ⊢ Eq (sigmaFinsuppEquivDFinsupp f).support f.splitSupport
  -/
  ext
  /-
    case h
    ι : Type u_1
    η : ι → Type u_4
    N : Type u_5
    inst✝² : DecidableEq ι
    inst✝¹ : Zero N
    inst✝ : (i : ι) → (x : Finsupp (η i) N) → Decidable (Ne x 0)
    f : Finsupp (Sigma fun i => η i) N
    a✝ : ι
    ⊢ Iff (Membership.mem (sigmaFinsuppEquivDFinsupp f).support a✝) (Membership.me …
  -/
  rw [DFinsupp.mem_support_toFun]
  /-
    case h
    ι : Type u_1
    η : ι → Type u_4
    N : Type u_5
    inst✝² : DecidableEq ι
    inst✝¹ : Zero N
    inst✝ : (i : ι) → (x : Finsupp (η i) N) → Decidable (Ne x 0)
    f : Finsupp (Sigma fun i => η i) N
    a✝ : ι
    ⊢ Iff (Ne ((sigmaFinsuppEquivDFinsupp f) a✝) 0) (Membership.mem f.splitSupport …
  -/
  exact (Finsupp.mem_splitSupport_iff_nonzero _ _).symm
  /-
    🎉 no goals
  -/


@[simp]
theorem sigmaFinsuppEquivDFinsupp_single [DecidableEq ι] [Zero N] (a : Σi, η i) (n : N) :
    sigmaFinsuppEquivDFinsupp (Finsupp.single a n) =
      @DFinsupp.single _ (fun i => η i →₀ N) _ _ a.1 (Finsupp.single a.2 n) := by
  /-
    ι : Type u_1
    η : ι → Type u_4
    N : Type u_5
    inst✝¹ : DecidableEq ι
    inst✝ : Zero N
    a : Sigma fun i => η i
    n : N
    ⊢ Eq (sigmaFinsuppEquivDFinsupp (Finsupp.single a n)) (DFinsupp.single a.fst ( …
  -/
  obtain ⟨i, a⟩ := a
  /-
    case mk
    ι : Type u_1
    η : ι → Type u_4
    N : Type u_5
    inst✝¹ : DecidableEq ι
    inst✝ : Zero N
    n : N
    i : ι
    a : η i
    ⊢ Eq (sigmaFinsuppEquivDFinsupp (Finsupp.single ⟨i, a⟩ n)) (DFinsupp.single ⟨i …
  -/
  ext j b
  /-
    case mk.h.h
    ι : Type u_1
    η : ι → Type u_4
    N : Type u_5
    inst✝¹ : DecidableEq ι
    inst✝ : Zero N
    n : N
    i : ι
    a : η i
    j : ι
    b : η j
    ⊢ Eq (((sigmaFinsuppEquivDFinsupp (Finsupp.single ⟨i, a⟩ n)) j) b) (((DFinsupp …
  -/
  by_cases h : i = j
    /-
      case pos
      ι : Type u_1
      η : ι → Type u_4
      N : Type u_5
      inst✝¹ : DecidableEq ι
      inst✝ : Zero N
      n : N
      i : ι
      a : η i
      j : ι
      b : η j
      h : Eq i j
      ⊢ Eq (((sigmaFinsuppEquivDFinsupp (Finsupp.single ⟨i, a⟩ n)) j) b) (((DFinsupp …
    -/
  · subst h
    /-
      case pos
      ι : Type u_1
      η : ι → Type u_4
      N : Type u_5
      inst✝¹ : DecidableEq ι
      inst✝ : Zero N
      n : N
      i : ι
      a b : η i
      ⊢ Eq (((sigmaFinsuppEquivDFinsupp (Finsupp.single ⟨i, a⟩ n)) i) b) (((DFinsupp …
    -/
    classical simp [split_apply, Finsupp.single_apply]
    /-
      🎉 no goals
    -/
  /-
    case neg
    ι : Type u_1
    η : ι → Type u_4
    N : Type u_5
    inst✝¹ : DecidableEq ι
    inst✝ : Zero N
    n : N
    i : ι
    a : η i
    j : ι
    b : η j
    h : Not (Eq i j)
    ⊢ Eq (((sigmaFinsuppEquivDFinsupp (Finsupp.single ⟨i, a⟩ n)) j) b) (((DFinsupp …
  -/
  suffices Finsupp.single (⟨i, a⟩ : Σi, η i) n ⟨j, b⟩ = 0 by simp [split_apply, dif_neg h, this]
  /-
    case neg
    ι : Type u_1
    η : ι → Type u_4
    N : Type u_5
    inst✝¹ : DecidableEq ι
    inst✝ : Zero N
    n : N
    i : ι
    a : η i
    j : ι
    b : η j
    h : Not (Eq i j)
    ⊢ Eq ((Finsupp.single ⟨i, a⟩ n) ⟨j, b⟩) 0
  -/
  have H : (⟨i, a⟩ : Σi, η i) ≠ ⟨j, b⟩ := by simp [h]
  /-
    case neg
    ι : Type u_1
    η : ι → Type u_4
    N : Type u_5
    inst✝¹ : DecidableEq ι
    inst✝ : Zero N
    n : N
    i : ι
    a : η i
    j : ι
    b : η j
    h : Not (Eq i j)
    H : Ne ⟨i, a⟩ ⟨j, b⟩
    ⊢ Eq ((Finsupp.single ⟨i, a⟩ n) ⟨j, b⟩) 0
  -/
  classical rw [Finsupp.single_apply, if_neg H]
  /-
    🎉 no goals
  -/

-- Without this Lean fails to find the `AddZeroClass` instance on `Π₀ i, (η i →₀ N)`.

@[simp]
theorem sigmaFinsuppEquivDFinsupp_add [AddZeroClass N] (f g : (Σi, η i) →₀ N) :
    sigmaFinsuppEquivDFinsupp (f + g) =
      (sigmaFinsuppEquivDFinsupp f + sigmaFinsuppEquivDFinsupp g : Π₀ i : ι, η i →₀ N) := by
  /-
    ι : Type u_1
    η : ι → Type u_4
    N : Type u_5
    inst✝ : AddZeroClass N
    f g : Finsupp (Sigma fun i => η i) N
    ⊢ Eq (sigmaFinsuppEquivDFinsupp (HAdd.hAdd f g)) (HAdd.hAdd (sigmaFinsuppEquiv …
  -/
  ext
  /-
    case h.h
    ι : Type u_1
    η : ι → Type u_4
    N : Type u_5
    inst✝ : AddZeroClass N
    f g : Finsupp (Sigma fun i => η i) N
    i✝ : ι
    a✝ : η i✝
    ⊢ Eq (((sigmaFinsuppEquivDFinsupp (HAdd.hAdd f g)) i✝) a✝) (((HAdd.hAdd (sigma …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- `Finsupp.split` is an additive equivalence between `(Σ i, η i) →₀ N` and `Π₀ i, (η i →₀ N)`. -/
@[simps]
def sigmaFinsuppAddEquivDFinsupp [AddZeroClass N] : ((Σi, η i) →₀ N) ≃+ Π₀ i, η i →₀ N :=
  { sigmaFinsuppEquivDFinsupp with
    toFun := sigmaFinsuppEquivDFinsupp
    invFun := sigmaFinsuppEquivDFinsupp.symm
    map_add' := sigmaFinsuppEquivDFinsupp_add }


@[simp]
theorem sigmaFinsuppEquivDFinsupp_smul {R} [Monoid R] [AddMonoid N] [DistribMulAction R N] (r : R)
    (f : (Σ i, η i) →₀ N) :
    sigmaFinsuppEquivDFinsupp (r • f) = r • sigmaFinsuppEquivDFinsupp f := by
  /-
    ι : Type u_1
    η : ι → Type u_4
    N : Type u_5
    R : Type u_6
    inst✝² : Monoid R
    inst✝¹ : AddMonoid N
    inst✝ : DistribMulAction R N
    r : R
    f : Finsupp (Sigma fun i => η i) N
    ⊢ Eq (sigmaFinsuppEquivDFinsupp (HSMul.hSMul r f)) (HSMul.hSMul r (sigmaFinsup …
  -/
  ext
  /-
    case h.h
    ι : Type u_1
    η : ι → Type u_4
    N : Type u_5
    R : Type u_6
    inst✝² : Monoid R
    inst✝¹ : AddMonoid N
    inst✝ : DistribMulAction R N
    r : R
    f : Finsupp (Sigma fun i => η i) N
    i✝ : ι
    a✝ : η i✝
    ⊢ Eq (((sigmaFinsuppEquivDFinsupp (HSMul.hSMul r f)) i✝) a✝) (((HSMul.hSMul r  …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- `Finsupp.split` is a linear equivalence between `(Σ i, η i) →₀ N` and `Π₀ i, (η i →₀ N)`. -/
@[simps]
def sigmaFinsuppLequivDFinsupp [AddCommMonoid N] [Module R N] :
    ((Σi, η i) →₀ N) ≃ₗ[R] Π₀ i, η i →₀ N :=
    -- Porting note: was
    -- sigmaFinsuppAddEquivDFinsupp with map_smul' := sigmaFinsuppEquivDFinsupp_smul
    -- but times out
  { sigmaFinsuppEquivDFinsupp with
    toFun := sigmaFinsuppEquivDFinsupp
    invFun := sigmaFinsuppEquivDFinsupp.symm
    map_add' := sigmaFinsuppEquivDFinsupp_add
    map_smul' := sigmaFinsuppEquivDFinsupp_smul }


