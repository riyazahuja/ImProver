local notation "surjective" => fun {X Y : Type _} [CommRing X] [CommRing Y] => fun f : X →+* Y =>
  Function.Surjective f


theorem surjective_stableUnderComposition : StableUnderComposition surjective := by
  /-
    ⊢ RingHom.StableUnderComposition fun {X Y} [CommRing X] [CommRing Y] f => Func …
  -/
  introv R hf hg; exact hg.comp hf
                  /-
                    🎉 no goals
                  -/


theorem surjective_respectsIso : RespectsIso surjective := by
  /-
    ⊢ RingHom.RespectsIso fun {X Y} [CommRing X] [CommRing Y] f => Function.Surjec …
  -/
  apply surjective_stableUnderComposition.respectsIso
  /-
    ⊢ ∀ {R S : Type u_1} [inst : CommRing R] [inst_1 : CommRing S] (e : RingEquiv  …
  -/
  intros _ _ _ _ e
  /-
    R✝ S✝ : Type u_1
    inst✝¹ : CommRing R✝
    inst✝ : CommRing S✝
    e : RingEquiv R✝ S✝
    ⊢ Function.Surjective ⇑e.toRingHom
  -/
  exact e.surjective
  /-
    🎉 no goals
  -/


theorem surjective_isStableUnderBaseChange : IsStableUnderBaseChange surjective := by
  /-
    ⊢ RingHom.IsStableUnderBaseChange fun {X Y} [CommRing X] [CommRing Y] f => Fun …
  -/
  refine IsStableUnderBaseChange.mk _ surjective_respectsIso ?_
  classical
  introv h x
  induction x with
  | zero => exact ⟨0, map_zero _⟩
  | tmul x y =>
    obtain ⟨y, rfl⟩ := h y; use y • x; dsimp
    rw [TensorProduct.smul_tmul, Algebra.algebraMap_eq_smul_one]
  | add x y ex ey => obtain ⟨⟨x, rfl⟩, ⟨y, rfl⟩⟩ := ex, ey; exact ⟨x + y, map_add _ x y⟩


/-- `M⁻¹R →+* M⁻¹S` is surjective if `R →+* S` is surjective. -/
theorem surjective_localizationPreserves :
    LocalizationPreserves surjective := by
  /-
    ⊢ RingHom.LocalizationPreserves fun {X Y} [CommRing X] [CommRing Y] f => Funct …
  -/
  introv R H x
  /-
    R S : Type u_1
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    f : RingHom R S
    M : Submonoid R
    R' S' : Type u_1
    inst✝⁵ : CommRing R'
    inst✝⁴ : CommRing S'
    inst✝³ : Algebra R R'
    inst✝² : Algebra S S'
    inst✝¹ : IsLocalization M R'
    inst✝ : IsLocalization (Submonoid.map f M) S'
    H : Function.Surjective ⇑f
    x : S'
    ⊢ Exists fun a => Eq ((IsLocalization.map S' f ⋯) a) x
  -/
  obtain ⟨x, ⟨_, s, hs, rfl⟩, rfl⟩ := IsLocalization.mk'_surjective (M.map f) x
  /-
    case intro.intro.mk.intro.intro
    R S : Type u_1
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    f : RingHom R S
    M : Submonoid R
    R' S' : Type u_1
    inst✝⁵ : CommRing R'
    inst✝⁴ : CommRing S'
    inst✝³ : Algebra R R'
    inst✝² : Algebra S S'
    inst✝¹ : IsLocalization M R'
    inst✝ : IsLocalization (Submonoid.map f M) S'
    H : Function.Surjective ⇑f
    x : S
    s : R
    hs : Membership.mem (↑M) s
    ⊢ Exists fun a => Eq ((IsLocalization.map S' f ⋯) a) (IsLocalization.mk' S' x  …
  -/
  obtain ⟨y, rfl⟩ := H x
  /-
    case intro.intro.mk.intro.intro.intro
    R S : Type u_1
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    f : RingHom R S
    M : Submonoid R
    R' S' : Type u_1
    inst✝⁵ : CommRing R'
    inst✝⁴ : CommRing S'
    inst✝³ : Algebra R R'
    inst✝² : Algebra S S'
    inst✝¹ : IsLocalization M R'
    inst✝ : IsLocalization (Submonoid.map f M) S'
    H : Function.Surjective ⇑f
    s : R
    hs : Membership.mem (↑M) s
    y : R
    ⊢ Exists fun a => Eq ((IsLocalization.map S' f ⋯) a) (IsLocalization.mk' S' (f …
  -/
  use IsLocalization.mk' R' y ⟨s, hs⟩
  /-
    case h
    R S : Type u_1
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    f : RingHom R S
    M : Submonoid R
    R' S' : Type u_1
    inst✝⁵ : CommRing R'
    inst✝⁴ : CommRing S'
    inst✝³ : Algebra R R'
    inst✝² : Algebra S S'
    inst✝¹ : IsLocalization M R'
    inst✝ : IsLocalization (Submonoid.map f M) S'
    H : Function.Surjective ⇑f
    s : R
    hs : Membership.mem (↑M) s
    y : R
    ⊢ Eq ((IsLocalization.map S' f ⋯) (IsLocalization.mk' R' y ⟨s, hs⟩)) (IsLocali …
  -/
  rw [IsLocalization.map_mk']
  /-
    🎉 no goals
  -/


/-- `R →+* S` is surjective if there exists a set `{ r }` that spans `R` such that
  `Rᵣ →+* Sᵣ` is surjective. -/
theorem surjective_ofLocalizationSpan : OfLocalizationSpan surjective := by
  /-
    ⊢ RingHom.OfLocalizationSpan fun {X Y} [CommRing X] [CommRing Y] f => Function …
  -/
  introv R e H
  /-
    R S : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set R
    e : Eq (Ideal.span s) Top.top
    H : ∀ (r : ↑s), (fun {X Y} [CommRing X] [CommRing Y] f => Function.Surjective  …
    ⊢ Function.Surjective ⇑f
  -/
  rw [← Set.range_eq_univ, Set.eq_univ_iff_forall]
  /-
    R S : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set R
    e : Eq (Ideal.span s) Top.top
    H : ∀ (r : ↑s), (fun {X Y} [CommRing X] [CommRing Y] f => Function.Surjective  …
    ⊢ ∀ (x : S), Membership.mem (Set.range ⇑f) x
  -/
  letI := f.toAlgebra
  /-
    R S : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set R
    e : Eq (Ideal.span s) Top.top
    H : ∀ (r : ↑s), (fun {X Y} [CommRing X] [CommRing Y] f => Function.Surjective  …
    this : Algebra R S := f.toAlgebra
    ⊢ ∀ (x : S), Membership.mem (Set.range ⇑f) x
  -/
  intro x
  apply Submodule.mem_of_span_eq_top_of_smul_pow_mem
    (LinearMap.range (Algebra.linearMap R S)) s e
  /-
    case H
    R S : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set R
    e : Eq (Ideal.span s) Top.top
    H : ∀ (r : ↑s), (fun {X Y} [CommRing X] [CommRing Y] f => Function.Surjective  …
    this : Algebra R S := f.toAlgebra
    x : S
    ⊢ ∀ (r : ↑s), Exists fun n => Membership.mem (LinearMap.range (Algebra.linearM …
  -/
  intro r
  /-
    case H
    R S : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set R
    e : Eq (Ideal.span s) Top.top
    H : ∀ (r : ↑s), (fun {X Y} [CommRing X] [CommRing Y] f => Function.Surjective  …
    this : Algebra R S := f.toAlgebra
    x : S
    r : ↑s
    ⊢ Exists fun n => Membership.mem (LinearMap.range (Algebra.linearMap R S)) (HS …
  -/
  obtain ⟨a, e'⟩ := H r (algebraMap _ _ x)
  /-
    case H.intro
    R S : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set R
    e : Eq (Ideal.span s) Top.top
    H : ∀ (r : ↑s), (fun {X Y} [CommRing X] [CommRing Y] f => Function.Surjective  …
    this : Algebra R S := f.toAlgebra
    x : S
    r : ↑s
    a : Localization.Away ↑r
    e' : Eq ((Localization.awayMap f ↑r) a) ((algebraMap S (Localization.Away (f ↑ …
    ⊢ Exists fun n => Membership.mem (LinearMap.range (Algebra.linearMap R S)) (HS …
  -/
  obtain ⟨b, ⟨_, n, rfl⟩, rfl⟩ := IsLocalization.mk'_surjective (Submonoid.powers (r : R)) a
  /-
    case H.intro.intro.intro.mk.intro
    R S : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set R
    e : Eq (Ideal.span s) Top.top
    H : ∀ (r : ↑s), (fun {X Y} [CommRing X] [CommRing Y] f => Function.Surjective  …
    this : Algebra R S := f.toAlgebra
    x : S
    r : ↑s
    b : R
    n : Nat
    e' : Eq ((Localization.awayMap f ↑r) (IsLocalization.mk' (Localization.Away ↑r …
    ⊢ Exists fun n => Membership.mem (LinearMap.range (Algebra.linearMap R S)) (HS …
  -/
  erw [IsLocalization.map_mk'] at e'
  /-
    case H.intro.intro.intro.mk.intro
    R S : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set R
    e : Eq (Ideal.span s) Top.top
    H : ∀ (r : ↑s), (fun {X Y} [CommRing X] [CommRing Y] f => Function.Surjective  …
    this : Algebra R S := f.toAlgebra
    x : S
    r : ↑s
    b : R
    n : Nat
    e' : Eq (IsLocalization.mk' (Localization.Away (f ↑r)) (f b) ⟨f ↑⟨(fun x => HP …
    ⊢ Exists fun n => Membership.mem (LinearMap.range (Algebra.linearMap R S)) (HS …
  -/
  rw [eq_comm, IsLocalization.eq_mk'_iff_mul_eq, Subtype.coe_mk, Subtype.coe_mk, ← map_mul] at e'
  /-
    case H.intro.intro.intro.mk.intro
    R S : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set R
    e : Eq (Ideal.span s) Top.top
    H : ∀ (r : ↑s), (fun {X Y} [CommRing X] [CommRing Y] f => Function.Surjective  …
    this : Algebra R S := f.toAlgebra
    x : S
    r : ↑s
    b : R
    n : Nat
    e' : Eq ((algebraMap S (Localization.Away (f ↑r))) (HMul.hMul x ↑⟨f ↑⟨(fun x = …
    ⊢ Exists fun n => Membership.mem (LinearMap.range (Algebra.linearMap R S)) (HS …
  -/
  obtain ⟨⟨_, n', rfl⟩, e''⟩ := (IsLocalization.eq_iff_exists (Submonoid.powers (f r)) _).mp e'
  /-
    case H.intro.intro.intro.mk.intro.intro.mk.intro
    R S : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set R
    e : Eq (Ideal.span s) Top.top
    H : ∀ (r : ↑s), (fun {X Y} [CommRing X] [CommRing Y] f => Function.Surjective  …
    this : Algebra R S := f.toAlgebra
    x : S
    r : ↑s
    b : R
    n : Nat
    e' : Eq ((algebraMap S (Localization.Away (f ↑r))) (HMul.hMul x ↑⟨f ↑⟨(fun x = …
    n' : Nat
    e'' : Eq (HMul.hMul (↑⟨(fun x => HPow.hPow (f ↑r) x) n', ⋯⟩) (HMul.hMul x ↑⟨f  …
    ⊢ Exists fun n => Membership.mem (LinearMap.range (Algebra.linearMap R S)) (HS …
  -/
  dsimp only at e''
  /-
    case H.intro.intro.intro.mk.intro.intro.mk.intro
    R S : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set R
    e : Eq (Ideal.span s) Top.top
    H : ∀ (r : ↑s), (fun {X Y} [CommRing X] [CommRing Y] f => Function.Surjective  …
    this : Algebra R S := f.toAlgebra
    x : S
    r : ↑s
    b : R
    n : Nat
    e' : Eq ((algebraMap S (Localization.Away (f ↑r))) (HMul.hMul x ↑⟨f ↑⟨(fun x = …
    n' : Nat
    e'' : Eq (HMul.hMul (HPow.hPow (f ↑r) n') (HMul.hMul x (f (HPow.hPow (↑r) n))) …
    ⊢ Exists fun n => Membership.mem (LinearMap.range (Algebra.linearMap R S)) (HS …
  -/
  rw [mul_comm x, ← mul_assoc, ← map_pow, ← map_mul, ← map_mul, ← pow_add] at e''
  /-
    case H.intro.intro.intro.mk.intro.intro.mk.intro
    R S : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set R
    e : Eq (Ideal.span s) Top.top
    H : ∀ (r : ↑s), (fun {X Y} [CommRing X] [CommRing Y] f => Function.Surjective  …
    this : Algebra R S := f.toAlgebra
    x : S
    r : ↑s
    b : R
    n : Nat
    e' : Eq ((algebraMap S (Localization.Away (f ↑r))) (HMul.hMul x ↑⟨f ↑⟨(fun x = …
    n' : Nat
    e'' : Eq (HMul.hMul (f (HPow.hPow (↑r) (HAdd.hAdd n' n))) x) (f (HMul.hMul (HP …
    ⊢ Exists fun n => Membership.mem (LinearMap.range (Algebra.linearMap R S)) (HS …
  -/
  exact ⟨n' + n, _, e''.symm⟩
  /-
    🎉 no goals
  -/


/-- A surjective ring homomorphism `R →+* S` induces a surjective homomorphism `R_{f⁻¹(P)} →+* S_P`
for every prime ideal `P` of `S`. -/
theorem surjective_localRingHom_of_surjective {R S : Type u} [CommRing R] [CommRing S]
    (f : R →+* S) (h : Function.Surjective f) (P : Ideal S) [P.IsPrime] :
    Function.Surjective (Localization.localRingHom (P.comap f) P f rfl) :=
  have : IsLocalization (Submonoid.map f (Ideal.comap f P).primeCompl) (Localization.AtPrime P) :=
    (Submonoid.map_comap_eq_of_surjective h P.primeCompl).symm ▸ Localization.isLocalization
  surjective_localizationPreserves _ _ _ _ h


