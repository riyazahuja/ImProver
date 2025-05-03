/-- A linear endomorphism of an `R`-module `M` is called *semisimple* if the induced `R[X]`-module
structure on `M` is semisimple. This is equivalent to saying that every `f`-invariant `R`-submodule
of `M` has an `f`-invariant complement: see `Module.End.isSemisimple_iff`. -/
def IsSemisimple := IsSemisimpleModule R[X] (AEval' f)


/-- A weaker version of semisimplicity that only prescribes behaviour on finitely-generated
submodules. -/
def IsFinitelySemisimple : Prop :=
  ∀ p (hp : p ∈ invtSubmodule f), Module.Finite R p → IsSemisimple (LinearMap.restrict f hp)


/-- A linear endomorphism is semisimple if every invariant submodule has in invariant complement.

See also `Module.End.isSemisimple_iff`. -/
lemma isSemisimple_iff' :
    f.IsSemisimple ↔ ∀ p : invtSubmodule f, ∃ q : invtSubmodule f, IsCompl p q := by
  rw [IsSemisimple, IsSemisimpleModule, (AEval.mapSubmodule R M f).symm.complementedLattice_iff,
    complementedLattice_iff]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    ⊢ Iff (∀ (a : Subtype fun x => Membership.mem ((Algebra.lsmul R R M) f).invtSu …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma isSemisimple_iff :
    f.IsSemisimple ↔ ∀ p ∈ invtSubmodule f, ∃ q ∈ invtSubmodule f, IsCompl p q := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    ⊢ Iff f.IsSemisimple (∀ (p : Submodule R M), Membership.mem f.invtSubmodule p  …
  -/
  simp_rw [isSemisimple_iff']
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    ⊢ Iff (∀ (p : Subtype fun x => Membership.mem f.invtSubmodule x), Exists fun q …
  -/
  aesop
  /-
    🎉 no goals
  -/


lemma isSemisimple_restrict_iff (p) (hp : p ∈ invtSubmodule f) :
    IsSemisimple (LinearMap.restrict f hp) ↔
    ∀ q ∈ f.invtSubmodule, q ≤ p → ∃ r ≤ p, r ∈ f.invtSubmodule ∧ Disjoint q r ∧ q ⊔ r = p := by
  let e : Submodule R[X] (AEval' (f.restrict hp)) ≃o Iic (AEval.mapSubmodule R M f ⟨p, hp⟩) :=
    (Submodule.orderIsoMapComap <| AEval.restrict_equiv_mapSubmodule f p hp).trans
      (Submodule.mapIic _)
  simp_rw [IsSemisimple, IsSemisimpleModule, e.complementedLattice_iff, disjoint_iff,
    ← (OrderIso.Iic _ _).complementedLattice_iff, Iic.complementedLattice_iff, Subtype.forall,
    Subtype.exists, Subtype.mk_le_mk, Sublattice.mk_inf_mk, Sublattice.mk_sup_mk, Subtype.mk.injEq,
    exists_and_left, exists_and_right, invtSubmodule.mk_eq_bot_iff, exists_prop, and_assoc]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    p : Submodule R M
    hp : Membership.mem f.invtSubmodule p
    e : OrderIso (Submodule (Polynomial R) (Module.AEval' (LinearMap.restrict f hp …
    ⊢ Iff (∀ (a : Submodule R M), Membership.mem ((Algebra.lsmul R R M) f).invtSub …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- A linear endomorphism is finitely semisimple if it is semisimple on every finitely-generated
invariant submodule.

See also `Module.End.isFinitelySemisimple_iff`. -/
lemma isFinitelySemisimple_iff' :
    f.IsFinitelySemisimple ↔ ∀ p (hp : p ∈ invtSubmodule f),
      Module.Finite R p → IsSemisimple (LinearMap.restrict f hp) :=
  Iff.rfl


/-- A characterisation of `Module.End.IsFinitelySemisimple` using only the lattice of submodules of
`M` (thus avoiding submodules of submodules). -/
lemma isFinitelySemisimple_iff :
    f.IsFinitelySemisimple ↔ ∀ p ∈ invtSubmodule f, Module.Finite R p → ∀ q ∈ invtSubmodule f,
      q ≤ p → ∃ r, r ≤ p ∧ r ∈ invtSubmodule f ∧ Disjoint q r ∧ q ⊔ r = p := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    ⊢ Iff f.IsFinitelySemisimple (∀ (p : Submodule R M), Membership.mem f.invtSubm …
  -/
  simp_rw [isFinitelySemisimple_iff', isSemisimple_restrict_iff]
  /-
    🎉 no goals
  -/


@[simp]
lemma isSemisimple_zero [IsSemisimpleModule R M] : IsSemisimple (0 : Module.End R M) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsSemisimpleModule R M
    ⊢ Module.End.IsSemisimple 0
  -/
  simpa [isSemisimple_iff] using exists_isCompl
  /-
    🎉 no goals
  -/


@[simp]
lemma isSemisimple_id [IsSemisimpleModule R M] : IsSemisimple (LinearMap.id : Module.End R M) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsSemisimpleModule R M
    ⊢ Module.End.IsSemisimple LinearMap.id
  -/
  simpa [isSemisimple_iff] using exists_isCompl
  /-
    🎉 no goals
  -/


@[simp] lemma isSemisimple_neg : (-f).IsSemisimple ↔ f.IsSemisimple := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    ⊢ Iff (Neg.neg f).IsSemisimple f.IsSemisimple
  -/
  simp [isSemisimple_iff, mem_invtSubmodule]
  /-
    🎉 no goals
  -/


variable (f) in
protected lemma _root_.LinearEquiv.isSemisimple_iff {M₂ : Type*} [AddCommGroup M₂] [Module R M₂]
    (g : End R M₂) (e : M ≃ₗ[R] M₂) (he : e ∘ₗ f = g ∘ₗ e) :
    f.IsSemisimple ↔ g.IsSemisimple := by
  let e : AEval' f ≃ₗ[R[X]] AEval' g := LinearEquiv.ofAEval _ (e.trans (AEval'.of g)) fun x ↦ by
    simpa [AEval'.X_smul_of] using LinearMap.congr_fun he x
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    f : Module.End R M
    M₂ : Type u_3
    inst✝¹ : AddCommGroup M₂
    inst✝ : Module R M₂
    g : Module.End R M₂
    e✝ : LinearEquiv (RingHom.id R) M M₂
    he : Eq ((↑e✝).comp f) (LinearMap.comp g ↑e✝)
    e : LinearEquiv (RingHom.id (Polynomial R)) (Module.AEval' f) (Module.AEval' g …
    ⊢ Iff f.IsSemisimple g.IsSemisimple
  -/
  exact (Submodule.orderIsoMapComap e).complementedLattice_iff
  /-
    🎉 no goals
  -/


lemma eq_zero_of_isNilpotent_isSemisimple (hn : IsNilpotent f) (hs : f.IsSemisimple) : f = 0 := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    hn : IsNilpotent f
    hs : f.IsSemisimple
    ⊢ Eq f 0
  -/
  have ⟨n, h0⟩ := hn
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    hn : IsNilpotent f
    hs : f.IsSemisimple
    n : Nat
    h0 : Eq (HPow.hPow f n) 0
    ⊢ Eq f 0
  -/
  rw [← aeval_X (R := R) f]; rw [← aeval_X_pow (R := R) f] at h0
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    hn : IsNilpotent f
    hs : f.IsSemisimple
    n : Nat
    h0 : Eq ((Polynomial.aeval f) (HPow.hPow Polynomial.X n)) 0
    ⊢ Eq ((Polynomial.aeval f) Polynomial.X) 0
  -/
  rw [← RingHom.mem_ker, ← AEval.annihilator_eq_ker_aeval (M := M)] at h0 ⊢
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    hn : IsNilpotent f
    hs : f.IsSemisimple
    n : Nat
    h0 : Membership.mem (Module.annihilator (Polynomial R) (Module.AEval R M f)) ( …
    ⊢ Membership.mem (Module.annihilator (Polynomial R) (Module.AEval R M f)) Poly …
  -/
  exact hs.annihilator_isRadical _ _ ⟨n, h0⟩
  /-
    🎉 no goals
  -/


lemma eq_zero_of_isNilpotent_of_isFinitelySemisimple
    (hn : IsNilpotent f) (hs : IsFinitelySemisimple f) : f = 0 := by
  have (p) (hp₁ : p ∈ f.invtSubmodule) (hp₂ : Module.Finite R p) : f.restrict hp₁ = 0 := by
    specialize hs p hp₁ hp₂
    replace hn : IsNilpotent (f.restrict hp₁) := isNilpotent.restrict hp₁ hn
    exact eq_zero_of_isNilpotent_isSemisimple hn hs
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    hn : IsNilpotent f
    hs : f.IsFinitelySemisimple
    this : ∀ (p : Submodule R M) (hp₁ : Membership.mem f.invtSubmodule p), Module. …
    ⊢ Eq f 0
  -/
  ext x
  /-
    case h
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    hn : IsNilpotent f
    hs : f.IsFinitelySemisimple
    this : ∀ (p : Submodule R M) (hp₁ : Membership.mem f.invtSubmodule p), Module. …
    x : M
    ⊢ Eq (f x) (0 x)
  -/
  obtain ⟨k : ℕ, hk : f ^ k = 0⟩ := hn
  /-
    case h.intro
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    hs : f.IsFinitelySemisimple
    this : ∀ (p : Submodule R M) (hp₁ : Membership.mem f.invtSubmodule p), Module. …
    x : M
    k : Nat
    hk : Eq (HPow.hPow f k) 0
    ⊢ Eq (f x) (0 x)
  -/
  let p := Submodule.span R {(f ^ i) x | (i : ℕ) (_ : i ≤ k)}
  have hp₁ : p ∈ f.invtSubmodule := by
    simp only [mem_invtSubmodule, p, Submodule.span_le]
    rintro - ⟨i, hi, rfl⟩
    apply Submodule.subset_span
    rcases lt_or_eq_of_le hi with hik | rfl
    · exact ⟨i + 1, hik, by simpa [LinearMap.pow_apply] using iterate_succ_apply' f i x⟩
    · exact ⟨i, by simp [hk]⟩
  have hp₂ : Module.Finite R p := by
    let g : ℕ → M := fun i ↦ (f ^ i) x
    have hg : {(f ^ i) x | (i : ℕ) (_ : i ≤ k)} = g '' Iic k := by ext; simp [g]
    exact Module.Finite.span_of_finite _ <| hg ▸ toFinite (g '' Iic k)
  simpa [LinearMap.restrict_apply, Subtype.ext_iff] using
    LinearMap.congr_fun (this p hp₁ hp₂) ⟨x, Submodule.subset_span ⟨0, k.zero_le, rfl⟩⟩


@[simp]
lemma isSemisimple_sub_algebraMap_iff {μ : R} :
    (f - algebraMap R (End R M) μ).IsSemisimple ↔ f.IsSemisimple := by
  suffices ∀ p : Submodule R M, p ≤ p.comap (f - algebraMap R (Module.End R M) μ) ↔ p ≤ p.comap f by
    simp [mem_invtSubmodule, isSemisimple_iff, this]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    ⊢ ∀ (p : Submodule R M), Iff (LE.le p (Submodule.comap (HSub.hSub f ((algebraM …
  -/
  refine fun p ↦ ⟨fun h x hx ↦ ?_, fun h x hx ↦ p.sub_mem (h hx) (p.smul_mem μ hx)⟩
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    p : Submodule R M
    h : LE.le p (Submodule.comap (HSub.hSub f ((algebraMap R (Module.End R M)) μ)) …
    x : M
    hx : Membership.mem p x
    ⊢ Membership.mem (Submodule.comap f p) x
  -/
  simpa using p.add_mem (h hx) (p.smul_mem μ hx)
  /-
    🎉 no goals
  -/


lemma IsSemisimple.restrict {p : Submodule R M} (hp : p ∈ f.invtSubmodule) (hf : f.IsSemisimple) :
    IsSemisimple (f.restrict hp) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    p : Submodule R M
    hp : Membership.mem f.invtSubmodule p
    hf : f.IsSemisimple
    ⊢ Module.End.IsSemisimple (LinearMap.restrict f hp)
  -/
  rw [IsSemisimple] at hf ⊢
  let e : Submodule R[X] (AEval' (LinearMap.restrict f hp)) ≃o
      Iic (AEval.mapSubmodule R M f ⟨p, hp⟩) :=
    (Submodule.orderIsoMapComap <| AEval.restrict_equiv_mapSubmodule f p hp).trans <|
      Submodule.mapIic _
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    p : Submodule R M
    hp : Membership.mem f.invtSubmodule p
    hf : IsSemisimpleModule (Polynomial R) (Module.AEval' f)
    e : OrderIso (Submodule (Polynomial R) (Module.AEval' (LinearMap.restrict f hp …
    ⊢ IsSemisimpleModule (Polynomial R) (Module.AEval' (LinearMap.restrict f hp))
  -/
  exact e.complementedLattice_iff.mpr inferInstance
  /-
    🎉 no goals
  -/


lemma IsSemisimple.isFinitelySemisimple (hf : f.IsSemisimple) :
    f.IsFinitelySemisimple :=
  isFinitelySemisimple_iff'.mp fun _ _ _ ↦ hf.restrict _


@[simp]
lemma isFinitelySemisimple_iff_isSemisimple [Module.Finite R M] :
    f.IsFinitelySemisimple ↔ f.IsSemisimple := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    f : Module.End R M
    inst✝ : Module.Finite R M
    ⊢ Iff f.IsFinitelySemisimple f.IsSemisimple
  -/
  refine ⟨fun hf ↦ isSemisimple_iff.mpr fun p hp ↦ ?_, IsSemisimple.isFinitelySemisimple⟩
  obtain ⟨q, -, hq₁, hq₂, hq₃⟩ :=
    isFinitelySemisimple_iff.mp hf ⊤ (invtSubmodule.top_mem f) inferInstance p hp le_top
  /-
    case intro.intro.intro.intro
    R : Type u_1
    M : Type u_2
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    f : Module.End R M
    inst✝ : Module.Finite R M
    hf : f.IsFinitelySemisimple
    p : Submodule R M
    hp : Membership.mem f.invtSubmodule p
    q : Submodule R M
    hq₁ : Membership.mem f.invtSubmodule q
    hq₂ : Disjoint p q
    hq₃ : Eq (Max.max p q) Top.top
    ⊢ Exists fun q => And (Membership.mem f.invtSubmodule q) (IsCompl p q)
  -/
  exact ⟨q, hq₁, hq₂, codisjoint_iff.mpr hq₃⟩
  /-
    🎉 no goals
  -/


@[simp]
lemma isFinitelySemisimple_sub_algebraMap_iff {μ : R} :
    (f - algebraMap R (End R M) μ).IsFinitelySemisimple ↔ f.IsFinitelySemisimple := by
  suffices ∀ p : Submodule R M, p ≤ p.comap (f - algebraMap R (Module.End R M) μ) ↔ p ≤ p.comap f by
    simp_rw [isFinitelySemisimple_iff, mem_invtSubmodule, this]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    ⊢ ∀ (p : Submodule R M), Iff (LE.le p (Submodule.comap (HSub.hSub f ((algebraM …
  -/
  refine fun p ↦ ⟨fun h x hx ↦ ?_, fun h x hx ↦ p.sub_mem (h hx) (p.smul_mem μ hx)⟩
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    p : Submodule R M
    h : LE.le p (Submodule.comap (HSub.hSub f ((algebraMap R (Module.End R M)) μ)) …
    x : M
    hx : Membership.mem p x
    ⊢ Membership.mem (Submodule.comap f p) x
  -/
  simpa using p.add_mem (h hx) (p.smul_mem μ hx)
  /-
    🎉 no goals
  -/


lemma IsFinitelySemisimple.restrict {p : Submodule R M} (hp : p ∈ f.invtSubmodule)
    (hf : f.IsFinitelySemisimple) :
    IsFinitelySemisimple (f.restrict hp) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    p : Submodule R M
    hp : Membership.mem f.invtSubmodule p
    hf : f.IsFinitelySemisimple
    ⊢ Module.End.IsFinitelySemisimple (LinearMap.restrict f hp)
  -/
  intro q hq₁ hq₂
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    p : Submodule R M
    hp : Membership.mem f.invtSubmodule p
    hf : f.IsFinitelySemisimple
    q : Submodule R (Subtype fun x => Membership.mem p x)
    hq₁ : Membership.mem (Module.End.invtSubmodule (LinearMap.restrict f hp)) q
    hq₂ : Module.Finite R (Subtype fun x => Membership.mem q x)
    ⊢ Module.End.IsSemisimple ((LinearMap.restrict f hp).restrict hq₁)
  -/
  have := invtSubmodule.map_subtype_mem_of_mem_invtSubmodule f hp hq₁
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    p : Submodule R M
    hp : Membership.mem f.invtSubmodule p
    hf : f.IsFinitelySemisimple
    q : Submodule R (Subtype fun x => Membership.mem p x)
    hq₁ : Membership.mem (Module.End.invtSubmodule (LinearMap.restrict f hp)) q
    hq₂ : Module.Finite R (Subtype fun x => Membership.mem q x)
    this : Membership.mem f.invtSubmodule (Submodule.map p.subtype q)
    ⊢ Module.End.IsSemisimple ((LinearMap.restrict f hp).restrict hq₁)
  -/
  let e : q ≃ₗ[R] q.map p.subtype := p.equivSubtypeMap q
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    p : Submodule R M
    hp : Membership.mem f.invtSubmodule p
    hf : f.IsFinitelySemisimple
    q : Submodule R (Subtype fun x => Membership.mem p x)
    hq₁ : Membership.mem (Module.End.invtSubmodule (LinearMap.restrict f hp)) q
    hq₂ : Module.Finite R (Subtype fun x => Membership.mem q x)
    this : Membership.mem f.invtSubmodule (Submodule.map p.subtype q)
    e : LinearEquiv (RingHom.id R) (Subtype fun x => Membership.mem q x) (Subtype  …
    ⊢ Module.End.IsSemisimple ((LinearMap.restrict f hp).restrict hq₁)
  -/
  rw [e.isSemisimple_iff ((LinearMap.restrict f hp).restrict hq₁) (LinearMap.restrict f this) rfl]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    p : Submodule R M
    hp : Membership.mem f.invtSubmodule p
    hf : f.IsFinitelySemisimple
    q : Submodule R (Subtype fun x => Membership.mem p x)
    hq₁ : Membership.mem (Module.End.invtSubmodule (LinearMap.restrict f hp)) q
    hq₂ : Module.Finite R (Subtype fun x => Membership.mem q x)
    this : Membership.mem f.invtSubmodule (Submodule.map p.subtype q)
    e : LinearEquiv (RingHom.id R) (Subtype fun x => Membership.mem q x) (Subtype  …
    ⊢ Module.End.IsSemisimple (LinearMap.restrict f this)
  -/
  exact hf _ this (Finite.map q p.subtype)
  /-
    🎉 no goals
  -/


lemma IsSemisimple_smul_iff {t : K} (ht : t ≠ 0) :
    (t • f).IsSemisimple ↔ f.IsSemisimple := by
  /-
    M : Type u_2
    inst✝² : AddCommGroup M
    K : Type u_3
    inst✝¹ : Field K
    inst✝ : Module K M
    f : Module.End K M
    t : K
    ht : Ne t 0
    ⊢ Iff (HSMul.hSMul t f).IsSemisimple f.IsSemisimple
  -/
  simp [isSemisimple_iff, mem_invtSubmodule, Submodule.comap_smul f (h := ht)]
  /-
    🎉 no goals
  -/


lemma IsSemisimple_smul (t : K) (h : f.IsSemisimple) :
    (t • f).IsSemisimple := by
  /-
    M : Type u_2
    inst✝² : AddCommGroup M
    K : Type u_3
    inst✝¹ : Field K
    inst✝ : Module K M
    f : Module.End K M
    t : K
    h : f.IsSemisimple
    ⊢ (HSMul.hSMul t f).IsSemisimple
  -/
  wlog ht : t ≠ 0; · simp [not_not.mp ht]
                     /-
                       🎉 no goals
                     -/
  /-
    M✝ : Type u_2
    inst✝⁵ : AddCommGroup M✝
    K✝ : Type u_3
    inst✝⁴ : Field K✝
    inst✝³ : Module K✝ M✝
    f✝ : Module.End K✝ M✝
    M : Type u_2
    inst✝² : AddCommGroup M
    K : Type u_3
    inst✝¹ : Field K
    inst✝ : Module K M
    f : Module.End K M
    t : K
    h : f.IsSemisimple
    ht : Ne t 0
    ⊢ (HSMul.hSMul t f).IsSemisimple
  -/
  rwa [IsSemisimple_smul_iff ht]
  /-
    🎉 no goals
  -/


theorem isSemisimple_of_squarefree_aeval_eq_zero {p : K[X]}
    (hp : Squarefree p) (hpf : aeval f p = 0) : f.IsSemisimple := by
  rw [← RingHom.mem_ker, ← AEval.annihilator_eq_ker_aeval (M := M), mem_annihilator,
      ← IsTorsionBy, ← isTorsionBySet_singleton_iff, isTorsionBySet_iff_is_torsion_by_span] at hpf
  /-
    M : Type u_2
    inst✝² : AddCommGroup M
    K : Type u_3
    inst✝¹ : Field K
    inst✝ : Module K M
    f : Module.End K M
    p : Polynomial K
    hp : Squarefree p
    hpf : Module.IsTorsionBySet (Polynomial K) (Module.AEval K M f) ↑(Ideal.span ( …
    ⊢ f.IsSemisimple
  -/
  let R := K[X] ⧸ Ideal.span {p}
  have : IsReduced R :=
    (Ideal.isRadical_iff_quotient_reduced _).mp (isRadical_iff_span_singleton.mp hp.isRadical)
  /-
    M : Type u_2
    inst✝² : AddCommGroup M
    K : Type u_3
    inst✝¹ : Field K
    inst✝ : Module K M
    f : Module.End K M
    p : Polynomial K
    hp : Squarefree p
    hpf : Module.IsTorsionBySet (Polynomial K) (Module.AEval K M f) ↑(Ideal.span ( …
    R : Type u_3 := HasQuotient.Quotient (Polynomial K) (Ideal.span (Singleton.sin …
    this : IsReduced R
    ⊢ f.IsSemisimple
  -/
  have : FiniteDimensional K R := (AdjoinRoot.powerBasis hp.ne_zero).finite
  /-
    M : Type u_2
    inst✝² : AddCommGroup M
    K : Type u_3
    inst✝¹ : Field K
    inst✝ : Module K M
    f : Module.End K M
    p : Polynomial K
    hp : Squarefree p
    hpf : Module.IsTorsionBySet (Polynomial K) (Module.AEval K M f) ↑(Ideal.span ( …
    R : Type u_3 := HasQuotient.Quotient (Polynomial K) (Ideal.span (Singleton.sin …
    this✝ : IsReduced R
    this : FiniteDimensional K R
    ⊢ f.IsSemisimple
  -/
  have : IsArtinianRing R := .of_finite K R
  /-
    M : Type u_2
    inst✝² : AddCommGroup M
    K : Type u_3
    inst✝¹ : Field K
    inst✝ : Module K M
    f : Module.End K M
    p : Polynomial K
    hp : Squarefree p
    hpf : Module.IsTorsionBySet (Polynomial K) (Module.AEval K M f) ↑(Ideal.span ( …
    R : Type u_3 := HasQuotient.Quotient (Polynomial K) (Ideal.span (Singleton.sin …
    this✝¹ : IsReduced R
    this✝ : FiniteDimensional K R
    this : IsArtinianRing R
    ⊢ f.IsSemisimple
  -/
  have : IsSemisimpleRing R := IsArtinianRing.isSemisimpleRing_of_isReduced R
  /-
    M : Type u_2
    inst✝² : AddCommGroup M
    K : Type u_3
    inst✝¹ : Field K
    inst✝ : Module K M
    f : Module.End K M
    p : Polynomial K
    hp : Squarefree p
    hpf : Module.IsTorsionBySet (Polynomial K) (Module.AEval K M f) ↑(Ideal.span ( …
    R : Type u_3 := HasQuotient.Quotient (Polynomial K) (Ideal.span (Singleton.sin …
    this✝² : IsReduced R
    this✝¹ : FiniteDimensional K R
    this✝ : IsArtinianRing R
    this : IsSemisimpleRing R
    ⊢ f.IsSemisimple
  -/
  letI : Module R (AEval' f) := Module.IsTorsionBySet.module hpf
  let e : AEval' f →ₛₗ[Ideal.Quotient.mk (Ideal.span {p})] AEval' f :=
    { AddMonoidHom.id _ with map_smul' := fun _ _ ↦ rfl }
  /-
    M : Type u_2
    inst✝² : AddCommGroup M
    K : Type u_3
    inst✝¹ : Field K
    inst✝ : Module K M
    f : Module.End K M
    p : Polynomial K
    hp : Squarefree p
    hpf : Module.IsTorsionBySet (Polynomial K) (Module.AEval K M f) ↑(Ideal.span ( …
    R : Type u_3 := HasQuotient.Quotient (Polynomial K) (Ideal.span (Singleton.sin …
    this✝³ : IsReduced R
    this✝² : FiniteDimensional K R
    this✝¹ : IsArtinianRing R
    this✝ : IsSemisimpleRing R
    this : Module R (Module.AEval' f) := hpf.module
    e : LinearMap (Ideal.Quotient.mk (Ideal.span (Singleton.singleton p))) (Module …
      let __src := AddMonoidHom.id (Module.AEval' f);
      { toFun := (↑__src).toFun, map_add' := ⋯, map_smul' := ⋯ }
    ⊢ f.IsSemisimple
  -/
  exact (e.isSemisimpleModule_iff_of_bijective bijective_id).mpr inferInstance
  /-
    🎉 no goals
  -/


/-- The minimal polynomial of a semisimple endomorphism is square free -/
theorem IsSemisimple.minpoly_squarefree : Squarefree (minpoly K f) :=
  IsRadical.squarefree (minpoly.ne_zero <| Algebra.IsIntegral.isIntegral _) <| by
    /-
      M : Type u_2
      inst✝³ : AddCommGroup M
      K : Type u_3
      inst✝² : Field K
      inst✝¹ : Module K M
      f : Module.End K M
      inst✝ : FiniteDimensional K M
      hf : f.IsSemisimple
      ⊢ IsRadical (minpoly K f)
    -/
    rw [isRadical_iff_span_singleton, span_minpoly_eq_annihilator]; exact hf.annihilator_isRadical
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


protected theorem IsSemisimple.aeval (p : K[X]) : (aeval f p).IsSemisimple :=
  let R := K[X] ⧸ Ideal.span {minpoly K f}
  have : Module.Finite K R :=
    (AdjoinRoot.powerBasis' <| minpoly.monic <| Algebra.IsIntegral.isIntegral f).finite
  have : IsReduced R := (Ideal.isRadical_iff_quotient_reduced _).mp <|
    span_minpoly_eq_annihilator K f ▸ hf.annihilator_isRadical
  isSemisimple_of_squarefree_aeval_eq_zero ((minpoly.isRadical K _).squarefree <|
    minpoly.ne_zero <| .of_finite K <| Ideal.Quotient.mkₐ K (.span {minpoly K f}) p) <| by
      rw [← Ideal.Quotient.liftₐ_comp (.span {minpoly K f}) (aeval f)
        fun a h ↦ by rwa [Ideal.span, ← minpoly.ker_aeval_eq_span_minpoly] at h, aeval_algHom,
        AlgHom.comp_apply, AlgHom.comp_apply, ← aeval_algHom_apply, minpoly.aeval, map_zero]


theorem IsSemisimple.of_mem_adjoin_singleton {a : End K M}
    (ha : a ∈ Algebra.adjoin K {f}) : a.IsSemisimple := by
  /-
    M : Type u_2
    inst✝³ : AddCommGroup M
    K : Type u_3
    inst✝² : Field K
    inst✝¹ : Module K M
    f : Module.End K M
    inst✝ : FiniteDimensional K M
    hf : f.IsSemisimple
    a : Module.End K M
    ha : Membership.mem (Algebra.adjoin K (Singleton.singleton f)) a
    ⊢ a.IsSemisimple
  -/
  rw [Algebra.adjoin_singleton_eq_range_aeval] at ha; obtain ⟨p, rfl⟩ := ha; exact .aeval hf _
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


protected theorem IsSemisimple.pow (n : ℕ) : (f ^ n).IsSemisimple :=
  .of_mem_adjoin_singleton hf (pow_mem (Algebra.self_mem_adjoin_singleton _ _) _)


attribute [local simp] Submodule.Quotient.quot_mk_eq_mk in
theorem IsSemisimple.of_mem_adjoin_pair {a : End K M} (ha : a ∈ Algebra.adjoin K {f, g}) :
    a.IsSemisimple := by
  /-
    M : Type u_2
    inst✝⁴ : AddCommGroup M
    K : Type u_3
    inst✝³ : Field K
    inst✝² : Module K M
    f g : Module.End K M
    inst✝¹ : FiniteDimensional K M
    inst✝ : PerfectField K
    comm : Commute f g
    hf : f.IsSemisimple
    hg : g.IsSemisimple
    a : Module.End K M
    ha : Membership.mem (Algebra.adjoin K (Insert.insert f (Singleton.singleton g) …
    ⊢ a.IsSemisimple
  -/
  let R := K[X] ⧸ Ideal.span {minpoly K f}
  /-
    M : Type u_2
    inst✝⁴ : AddCommGroup M
    K : Type u_3
    inst✝³ : Field K
    inst✝² : Module K M
    f g : Module.End K M
    inst✝¹ : FiniteDimensional K M
    inst✝ : PerfectField K
    comm : Commute f g
    hf : f.IsSemisimple
    hg : g.IsSemisimple
    a : Module.End K M
    ha : Membership.mem (Algebra.adjoin K (Insert.insert f (Singleton.singleton g) …
    R : Type u_3 := HasQuotient.Quotient (Polynomial K) (Ideal.span (Singleton.sin …
    ⊢ a.IsSemisimple
  -/
  let S := AdjoinRoot ((minpoly K g).map <| algebraMap K R)
  have : Module.Finite K R :=
    (AdjoinRoot.powerBasis' <| minpoly.monic <| Algebra.IsIntegral.isIntegral f).finite
  have : Module.Finite R S :=
    (AdjoinRoot.powerBasis' <| (minpoly.monic <| Algebra.IsIntegral.isIntegral g).map _).finite
  #adaptation_note
  /--
  After https://github.com/leanprover/lean4/pull/4119 we either need
  to specify the `(S := R)` argument, or use `set_option maxSynthPendingDepth 2 in`.

  In either case this step is too slow!
  -/
  set_option maxSynthPendingDepth 2 in
  have : IsScalarTower K R S := .of_algebraMap_eq fun _ ↦ rfl
  have : Module.Finite K S := .trans R S
  have : IsArtinianRing R := .of_finite K R
  have : IsReduced R := (Ideal.isRadical_iff_quotient_reduced _).mp <|
    span_minpoly_eq_annihilator K f ▸ hf.annihilator_isRadical
  have : IsReduced S := by
    simp_rw [S, AdjoinRoot, ← Ideal.isRadical_iff_quotient_reduced, ← isRadical_iff_span_singleton]
    exact (PerfectField.separable_iff_squarefree.mpr hg.minpoly_squarefree).map.squarefree.isRadical
  let φ : S →ₐ[K] End K M := Ideal.Quotient.liftₐ _ (eval₂AlgHom' (Ideal.Quotient.liftₐ _ (aeval f)
    fun a ↦ ?_) g ?_) ((Ideal.span_singleton_le_iff_mem _).mpr ?_ : _ ≤ RingHom.ker _)
  rotate_left 1
  · rw [Ideal.span, ← minpoly.ker_aeval_eq_span_minpoly]; exact id
  · rintro ⟨p⟩; exact p.induction_on (fun k ↦ by simp [R, Algebra.commute_algebraMap_left])
      (fun p q hp hq ↦ by simpa [R] using hp.add_left hq)
      fun n k ↦ by simpa [R, pow_succ, ← mul_assoc _ _ X] using (·.mul_left comm)
  · simpa only [RingHom.mem_ker, eval₂AlgHom'_apply, eval₂_map, AlgHom.comp_algebraMap_of_tower]
      using minpoly.aeval K g
  have : Algebra.adjoin K {f, g} ≤ φ.range := Algebra.adjoin_le fun x ↦ by
    rintro (hx | hx) <;> rw [hx]
    · exact ⟨AdjoinRoot.of _ (AdjoinRoot.root _), (eval₂_C _ _).trans (aeval_X f)⟩
    · exact ⟨AdjoinRoot.root _, eval₂_X _ _⟩
  obtain ⟨p, rfl⟩ := (AlgHom.mem_range _).mp (this ha)
  refine isSemisimple_of_squarefree_aeval_eq_zero
    ((minpoly.isRadical K p).squarefree <| minpoly.ne_zero <| .of_finite K p) ?_
  rw [aeval_algHom, φ.comp_apply, minpoly.aeval, map_zero]


theorem IsSemisimple.add_of_commute : (f + g).IsSemisimple := .of_mem_adjoin_pair
  comm hf hg <| add_mem (Algebra.subset_adjoin <| .inl rfl) (Algebra.subset_adjoin <| .inr rfl)


theorem IsSemisimple.sub_of_commute : (f - g).IsSemisimple := .of_mem_adjoin_pair
  comm hf hg <| sub_mem (Algebra.subset_adjoin <| .inl rfl) (Algebra.subset_adjoin <| .inr rfl)


theorem IsSemisimple.mul_of_commute : (f * g).IsSemisimple := .of_mem_adjoin_pair
  comm hf hg <| mul_mem (Algebra.subset_adjoin <| .inl rfl) (Algebra.subset_adjoin <| .inr rfl)


