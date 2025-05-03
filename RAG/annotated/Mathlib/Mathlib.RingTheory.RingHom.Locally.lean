/--
For a property of ring homomorphisms `P`, `Locally P` holds for `f : R →+* S` if
it holds locally on `S`, i.e. if there exists a subset `{ t }` of `S` generating
the unit ideal, such that `P` holds for all compositions `R →+* Sₜ`.

We may require `s` to be finite here, for the equivalence, see `locally_iff_finite`.
-/
def Locally {R S : Type u} [CommRing R] [CommRing S] (f : R →+* S) : Prop :=
  ∃ (s : Set S) (_ : Ideal.span s = ⊤),
    ∀ t ∈ s, P ((algebraMap S (Localization.Away t)).comp f)


lemma locally_iff_finite (f : R →+* S) :
    Locally P f ↔ ∃ (s : Finset S) (_ : Ideal.span (s : Set S) = ⊤),
      ∀ t ∈ s, P ((algebraMap S (Localization.Away t)).comp f) := by
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    R S : Type u
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    ⊢ Iff (RingHom.Locally (fun {R S} [CommRing R] [CommRing S] => P) f) (Exists f …
  -/
  constructor
    /-
      case mp
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      R S : Type u
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      ⊢ RingHom.Locally (fun {R S} [CommRing R] [CommRing S] => P) f → Exists fun s  …
    -/
  · intro ⟨s, hsone, hs⟩
    /-
      case mp
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      R S : Type u
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      s : Set S
      hsone : Eq (Ideal.span s) Top.top
      hs : ∀ (t : S), Membership.mem s t → (fun {R S} [CommRing R] [CommRing S] => P …
      ⊢ Exists fun s => Exists fun x => ∀ (t : S), Membership.mem s t → P ((algebraM …
    -/
    obtain ⟨s', h₁, h₂⟩ := (Ideal.span_eq_top_iff_finite s).mp hsone
    /-
      case mp.intro.intro
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      R S : Type u
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      s : Set S
      hsone : Eq (Ideal.span s) Top.top
      hs : ∀ (t : S), Membership.mem s t → (fun {R S} [CommRing R] [CommRing S] => P …
      s' : Finset S
      h₁ : HasSubset.Subset (↑s') s
      h₂ : Eq (Ideal.span ↑s') Top.top
      ⊢ Exists fun s => Exists fun x => ∀ (t : S), Membership.mem s t → P ((algebraM …
    -/
    exact ⟨s', h₂, fun t ht ↦ hs t (h₁ ht)⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      R S : Type u
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      ⊢ (Exists fun s => Exists fun x => ∀ (t : S), Membership.mem s t → P ((algebra …
    -/
  · intro ⟨s, hsone, hs⟩
    /-
      case mpr
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      R S : Type u
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      s : Finset S
      hsone : Eq (Ideal.span ↑s) Top.top
      hs : ∀ (t : S), Membership.mem s t → P ((algebraMap S (Localization.Away t)).c …
      ⊢ RingHom.Locally (fun {R S} [CommRing R] [CommRing S] => P) f
    -/
    use s, hsone, hs
    /-
      🎉 no goals
    -/


/-- If `P` respects isomorphisms, to check `P` holds locally for `f : R →+* S`, it suffices
to check `P` holds on a standard open cover. -/
lemma locally_of_exists (hP : RespectsIso P) (f : R →+* S) {ι : Type*} (s : ι → S)
    (hsone : Ideal.span (Set.range s) = ⊤)
    (Sₜ : ι → Type u) [∀ i, CommRing (Sₜ i)] [∀ i, Algebra S (Sₜ i)]
    [∀ i, IsLocalization.Away (s i) (Sₜ i)] (hf : ∀ i, P ((algebraMap S (Sₜ i)).comp f)) :
    Locally P f := by
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    R S : Type u
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
    f : RingHom R S
    ι : Type u_1
    s : ι → S
    hsone : Eq (Ideal.span (Set.range s)) Top.top
    Sₜ : ι → Type u
    inst✝² : (i : ι) → CommRing (Sₜ i)
    inst✝¹ : (i : ι) → Algebra S (Sₜ i)
    inst✝ : ∀ (i : ι), IsLocalization.Away (s i) (Sₜ i)
    hf : ∀ (i : ι), P ((algebraMap S (Sₜ i)).comp f)
    ⊢ RingHom.Locally (fun {R S} [CommRing R] [CommRing S] => P) f
  -/
  use Set.range s, hsone
  /-
    case h
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    R S : Type u
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
    f : RingHom R S
    ι : Type u_1
    s : ι → S
    hsone : Eq (Ideal.span (Set.range s)) Top.top
    Sₜ : ι → Type u
    inst✝² : (i : ι) → CommRing (Sₜ i)
    inst✝¹ : (i : ι) → Algebra S (Sₜ i)
    inst✝ : ∀ (i : ι), IsLocalization.Away (s i) (Sₜ i)
    hf : ∀ (i : ι), P ((algebraMap S (Sₜ i)).comp f)
    ⊢ ∀ (t : S), Membership.mem (Set.range s) t → (fun {R S} [CommRing R] [CommRin …
  -/
  rintro - ⟨i, rfl⟩
  let e : Localization.Away (s i) ≃+* Sₜ i :=
    (IsLocalization.algEquiv (Submonoid.powers (s i)) _ _).toRingEquiv
  have : algebraMap S (Localization.Away (s i)) = e.symm.toRingHom.comp (algebraMap S (Sₜ i)) :=
    RingHom.ext (fun x ↦ (AlgEquiv.commutes (IsLocalization.algEquiv _ _ _).symm _).symm)
  /-
    case h.intro
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    R S : Type u
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
    f : RingHom R S
    ι : Type u_1
    s : ι → S
    hsone : Eq (Ideal.span (Set.range s)) Top.top
    Sₜ : ι → Type u
    inst✝² : (i : ι) → CommRing (Sₜ i)
    inst✝¹ : (i : ι) → Algebra S (Sₜ i)
    inst✝ : ∀ (i : ι), IsLocalization.Away (s i) (Sₜ i)
    hf : ∀ (i : ι), P ((algebraMap S (Sₜ i)).comp f)
    i : ι
    e : RingEquiv (Localization.Away (s i)) (Sₜ i) := (IsLocalization.algEquiv (Su …
    this : Eq (algebraMap S (Localization.Away (s i))) (e.symm.toRingHom.comp (alg …
    ⊢ P ((algebraMap S (Localization.Away (s i))).comp f)
  -/
  rw [this, RingHom.comp_assoc]
  /-
    case h.intro
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    R S : Type u
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
    f : RingHom R S
    ι : Type u_1
    s : ι → S
    hsone : Eq (Ideal.span (Set.range s)) Top.top
    Sₜ : ι → Type u
    inst✝² : (i : ι) → CommRing (Sₜ i)
    inst✝¹ : (i : ι) → Algebra S (Sₜ i)
    inst✝ : ∀ (i : ι), IsLocalization.Away (s i) (Sₜ i)
    hf : ∀ (i : ι), P ((algebraMap S (Sₜ i)).comp f)
    i : ι
    e : RingEquiv (Localization.Away (s i)) (Sₜ i) := (IsLocalization.algEquiv (Su …
    this : Eq (algebraMap S (Localization.Away (s i))) (e.symm.toRingHom.comp (alg …
    ⊢ P (e.symm.toRingHom.comp ((algebraMap S (Sₜ i)).comp f))
  -/
  exact hP.left _ _ (hf i)
  /-
    🎉 no goals
  -/


/-- Equivalence variant of `locally_of_exists`. This is sometimes easier to use, if the
`IsLocalization.Away` instance can't be automatically inferred. -/
lemma locally_iff_exists (hP : RespectsIso P) (f : R →+* S) :
    Locally P f ↔ ∃ (ι : Type u) (s : ι → S) (_ : Ideal.span (Set.range s) = ⊤) (Sₜ : ι → Type u)
      (_ : (i : ι) → CommRing (Sₜ i)) (_ : (i : ι) → Algebra S (Sₜ i))
      (_ : (i : ι) → IsLocalization.Away (s i : S) (Sₜ i)),
      ∀ i, P ((algebraMap S (Sₜ i)).comp f) :=
                                                    /-
                                                      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
                                                      R S : Type u
                                                      inst✝¹ : CommRing R
                                                      inst✝ : CommRing S
                                                      hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
                                                      f : RingHom R S
                                                      x✝ : RingHom.Locally (fun {R S} [CommRing R] [CommRing S] => P) f
                                                      s : Set S
                                                      hsone : Eq (Ideal.span s) Top.top
                                                      hs : ∀ (t : S), Membership.mem s t → (fun {R S} [CommRing R] [CommRing S] => P …
                                                      ⊢ Eq (Ideal.span (Set.range fun t => ↑t)) Top.top
                                                    -/
  ⟨fun ⟨s, hsone, hs⟩ ↦ ⟨s, fun t : s ↦ (t : S), by simpa, fun t ↦ Localization.Away (t : S),
                                                    /-
                                                      🎉 no goals
                                                    -/
      inferInstance, inferInstance, inferInstance, fun t ↦ hs t.val t.property⟩,
    fun ⟨ι, s, hsone, Sₜ, _, _, hislocal, hs⟩ ↦ locally_of_exists hP f s hsone Sₜ hs⟩


/-- In the definition of `Locally` we may replace `Localization.Away` with an arbitrary
algebra satisfying `IsLocalization.Away`. -/
lemma locally_iff_isLocalization (hP : RespectsIso P) (f : R →+* S) :
    Locally P f ↔ ∃ (s : Finset S) (_ : Ideal.span (s : Set S) = ⊤),
      ∀ t ∈ s, ∀ (Sₜ : Type u) [CommRing Sₜ] [Algebra S Sₜ] [IsLocalization.Away t Sₜ],
      P ((algebraMap S Sₜ).comp f) := by
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    R S : Type u
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
    f : RingHom R S
    ⊢ Iff (RingHom.Locally (fun {R S} [CommRing R] [CommRing S] => P) f) (Exists f …
  -/
  rw [locally_iff_finite P f]
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    R S : Type u
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
    f : RingHom R S
    ⊢ Iff (Exists fun s => Exists fun x => ∀ (t : S), Membership.mem s t → P ((alg …
  -/
  refine ⟨fun ⟨s, hsone, hs⟩ ↦ ⟨s, hsone, fun t ht Sₜ _ _ _ ↦ ?_⟩, fun ⟨s, hsone, hs⟩ ↦ ?_⟩
  · let e : Localization.Away t ≃+* Sₜ :=
      (IsLocalization.algEquiv (Submonoid.powers t) _ _).toRingEquiv
    have : algebraMap S Sₜ = e.toRingHom.comp (algebraMap S (Localization.Away t)) :=
      RingHom.ext (fun x ↦ (AlgEquiv.commutes (IsLocalization.algEquiv _ _ _) _).symm)
    /-
      case refine_1
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      R S : Type u
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      f : RingHom R S
      x✝³ : Exists fun s => Exists fun x => ∀ (t : S), Membership.mem s t → P ((alge …
      s : Finset S
      hsone : Eq (Ideal.span ↑s) Top.top
      hs : ∀ (t : S), Membership.mem s t → P ((algebraMap S (Localization.Away t)).c …
      t : S
      ht : Membership.mem s t
      Sₜ : Type u
      x✝² : CommRing Sₜ
      x✝¹ : Algebra S Sₜ
      x✝ : IsLocalization.Away t Sₜ
      e : RingEquiv (Localization.Away t) Sₜ := (IsLocalization.algEquiv (Submonoid. …
      this : Eq (algebraMap S Sₜ) (e.toRingHom.comp (algebraMap S (Localization.Away …
      ⊢ P ((algebraMap S Sₜ).comp f)
    -/
    rw [this, RingHom.comp_assoc]
    /-
      case refine_1
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      R S : Type u
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      f : RingHom R S
      x✝³ : Exists fun s => Exists fun x => ∀ (t : S), Membership.mem s t → P ((alge …
      s : Finset S
      hsone : Eq (Ideal.span ↑s) Top.top
      hs : ∀ (t : S), Membership.mem s t → P ((algebraMap S (Localization.Away t)).c …
      t : S
      ht : Membership.mem s t
      Sₜ : Type u
      x✝² : CommRing Sₜ
      x✝¹ : Algebra S Sₜ
      x✝ : IsLocalization.Away t Sₜ
      e : RingEquiv (Localization.Away t) Sₜ := (IsLocalization.algEquiv (Submonoid. …
      this : Eq (algebraMap S Sₜ) (e.toRingHom.comp (algebraMap S (Localization.Away …
      ⊢ P (e.toRingHom.comp ((algebraMap S (Localization.Away t)).comp f))
    -/
    exact hP.left _ _ (hs t ht)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      R S : Type u
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      f : RingHom R S
      x✝ : Exists fun s => Exists fun x => ∀ (t : S), Membership.mem s t → ∀ (Sₜ : T …
      s : Finset S
      hsone : Eq (Ideal.span ↑s) Top.top
      hs : ∀ (t : S), Membership.mem s t → ∀ (Sₜ : Type u) [inst : CommRing Sₜ] [ins …
      ⊢ Exists fun s => Exists fun x => ∀ (t : S), Membership.mem s t → P ((algebraM …
    -/
  · exact ⟨s, hsone, fun t ht ↦ hs t ht _⟩
    /-
      🎉 no goals
    -/


/-- If `f` satisfies `P`, then in particular it satisfies `Locally P`. -/
lemma locally_of (hP : RespectsIso P) (f : R →+* S) (hf : P f) : Locally P f := by
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    R S : Type u
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
    f : RingHom R S
    hf : P f
    ⊢ RingHom.Locally (fun {R S} [CommRing R] [CommRing S] => P) f
  -/
  use {1}
  let e : S ≃+* Localization.Away (1 : S) :=
    (IsLocalization.atUnits S (Submonoid.powers 1) (by simp)).toRingEquiv
  /-
    case h
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    R S : Type u
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
    f : RingHom R S
    hf : P f
    e : RingEquiv S (Localization.Away 1) := (IsLocalization.atUnits S (Submonoid. …
    ⊢ Exists fun x => ∀ (t : S), Membership.mem (Singleton.singleton 1) t → (fun { …
  -/
  simp only [Set.mem_singleton_iff, forall_eq, Ideal.span_singleton_one, exists_const]
  /-
    case h
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    R S : Type u
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
    f : RingHom R S
    hf : P f
    e : RingEquiv S (Localization.Away 1) := (IsLocalization.atUnits S (Submonoid. …
    ⊢ P ((algebraMap S (Localization.Away 1)).comp f)
  -/
  exact hP.left f e hf
  /-
    🎉 no goals
  -/


lemma locally_of_locally {Q : ∀ {R S : Type u} [CommRing R] [CommRing S], (R →+* S) → Prop}
    (hPQ : ∀ {R S : Type u} [CommRing R] [CommRing S] {f : R →+* S}, P f → Q f)
    {R S : Type u} [CommRing R] [CommRing S] {f : R →+* S} (hf : Locally P f) : Locally Q f := by
  /-
    P Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R …
    hPQ : ∀ {R S : Type u} [inst : CommRing R] [inst_1 : CommRing S] {f : RingHom  …
    R S : Type u
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    hf : RingHom.Locally (fun {R S} [CommRing R] [CommRing S] => P) f
    ⊢ RingHom.Locally (fun {R S} [CommRing R] [CommRing S] => Q) f
  -/
  obtain ⟨s, hsone, hs⟩ := hf
  /-
    case intro.intro
    P Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R …
    hPQ : ∀ {R S : Type u} [inst : CommRing R] [inst_1 : CommRing S] {f : RingHom  …
    R S : Type u
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set S
    hsone : Eq (Ideal.span s) Top.top
    hs : ∀ (t : S), Membership.mem s t → (fun {R S} [CommRing R] [CommRing S] => P …
    ⊢ RingHom.Locally (fun {R S} [CommRing R] [CommRing S] => Q) f
  -/
  exact ⟨s, hsone, fun t ht ↦ hPQ (hs t ht)⟩
  /-
    🎉 no goals
  -/


/-- If `P` is local on the target, then `Locally P` coincides with `P`. -/
lemma locally_iff_of_localizationSpanTarget (hPi : RespectsIso P)
    (hPs : OfLocalizationSpanTarget P) {R S : Type u} [CommRing R] [CommRing S] (f : R →+* S) :
    Locally P f ↔ P f :=
  ⟨fun ⟨s, hsone, hs⟩ ↦ hPs f s hsone (fun a ↦ hs a.val a.property), locally_of hPi f⟩


/-- `Locally P` is local on the target. -/
lemma locally_ofLocalizationSpanTarget (hP : RespectsIso P) :
    OfLocalizationSpanTarget (Locally P) := by
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
    ⊢ RingHom.OfLocalizationSpanTarget fun {R S} [CommRing R] [CommRing S] => Ring …
  -/
  intro R S _ _ f s hsone hs
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
    R S : Type u
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set S
    hsone : Eq (Ideal.span s) Top.top
    hs : ∀ (r : ↑s), (fun {R S} [CommRing R] [CommRing S] => RingHom.Locally fun { …
    ⊢ RingHom.Locally (fun {R S} [CommRing R] [CommRing S] => P) f
  -/
  choose t htone ht using hs
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
    R S : Type u
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set S
    hsone : Eq (Ideal.span s) Top.top
    t : (r : ↑s) → Set (Localization.Away ↑r)
    htone : ∀ (r : ↑s), Eq (Ideal.span (t r)) Top.top
    ht : ∀ (r : ↑s) (t_1 : Localization.Away ↑r), Membership.mem (t r) t_1 → (fun  …
    ⊢ RingHom.Locally (fun {R S} [CommRing R] [CommRing S] => P) f
  -/
  rw [locally_iff_exists hP]
  refine ⟨(a : s) × t a, IsLocalization.Away.mulNumerator s t,
      IsLocalization.Away.span_range_mulNumerator_eq_top hsone htone,
      fun ⟨a, b⟩ ↦ Localization.Away b.val, inferInstance, inferInstance, fun ⟨a, b⟩ ↦ ?_, ?_⟩
  · haveI : IsLocalization.Away ((algebraMap S (Localization.Away a.val))
        (IsLocalization.Away.sec a.val b.val).1) (Localization.Away b.val) := by
      apply IsLocalization.Away.of_associated (r := b.val)
      rw [← IsLocalization.Away.sec_spec]
      apply associated_mul_unit_right
      rw [map_pow _ _]
      exact IsUnit.pow _ (IsLocalization.Away.algebraMap_isUnit _)
    /-
      case refine_1
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      R S : Type u
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      s : Set S
      hsone : Eq (Ideal.span s) Top.top
      t : (r : ↑s) → Set (Localization.Away ↑r)
      htone : ∀ (r : ↑s), Eq (Ideal.span (t r)) Top.top
      ht : ∀ (r : ↑s) (t_1 : Localization.Away ↑r), Membership.mem (t r) t_1 → (fun  …
      x✝ : Sigma fun a => ↑(t a)
      a : ↑s
      b : ↑(t a)
      this : IsLocalization.Away ((algebraMap S (Localization.Away ↑a)) (IsLocalizat …
      ⊢ IsLocalization.Away (IsLocalization.Away.mulNumerator s t ⟨a, b⟩) ((fun x => …
    -/
    apply IsLocalization.Away.mul' (Localization.Away a.val) (Localization.Away b.val)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      R S : Type u
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      s : Set S
      hsone : Eq (Ideal.span s) Top.top
      t : (r : ↑s) → Set (Localization.Away ↑r)
      htone : ∀ (r : ↑s), Eq (Ideal.span (t r)) Top.top
      ht : ∀ (r : ↑s) (t_1 : Localization.Away ↑r), Membership.mem (t r) t_1 → (fun  …
      ⊢ ∀ (i : Sigma fun a => ↑(t a)), P ((algebraMap S ((fun x => RingHom.locally_o …
    -/
  · intro ⟨a, b⟩
    /-
      case refine_2
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      R S : Type u
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      s : Set S
      hsone : Eq (Ideal.span s) Top.top
      t : (r : ↑s) → Set (Localization.Away ↑r)
      htone : ∀ (r : ↑s), Eq (Ideal.span (t r)) Top.top
      ht : ∀ (r : ↑s) (t_1 : Localization.Away ↑r), Membership.mem (t r) t_1 → (fun  …
      a : ↑s
      b : ↑(t a)
      ⊢ P ((algebraMap S ((fun x => RingHom.locally_ofLocalizationSpanTarget.match_1 …
    -/
    rw [IsScalarTower.algebraMap_eq S (Localization.Away a.val) (Localization.Away b.val)]
    /-
      case refine_2
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      R S : Type u
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      s : Set S
      hsone : Eq (Ideal.span s) Top.top
      t : (r : ↑s) → Set (Localization.Away ↑r)
      htone : ∀ (r : ↑s), Eq (Ideal.span (t r)) Top.top
      ht : ∀ (r : ↑s) (t_1 : Localization.Away ↑r), Membership.mem (t r) t_1 → (fun  …
      a : ↑s
      b : ↑(t a)
      ⊢ P (((algebraMap (Localization.Away ↑a) (Localization.Away ↑b)).comp (algebra …
    -/
    apply ht _ _ b.property
    /-
      🎉 no goals
    -/


/-- If `P` respects isomorphism, so does `Locally P`. -/
lemma locally_respectsIso (hPi : RespectsIso P) : RespectsIso (Locally P) where
  left {R S T} _ _ _ f e := fun ⟨s, hsone, hs⟩ ↦ by
    /-
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      R S T : Type u
      x✝³ : CommRing R
      x✝² : CommRing S
      x✝¹ : CommRing T
      f : RingHom R S
      e : RingEquiv S T
      x✝ : (fun {R S} [CommRing R] [CommRing S] => RingHom.Locally fun {R S} [CommRi …
      s : Set S
      hsone : Eq (Ideal.span s) Top.top
      hs : ∀ (t : S), Membership.mem s t → (fun {R S} [CommRing R] [CommRing S] => P …
      ⊢ (fun {R S} [CommRing R] [CommRing S] => RingHom.Locally fun {R S} [CommRing  …
    -/
    refine ⟨e '' s, ?_, ?_⟩
      /-
        case refine_1
        P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
        R S T : Type u
        x✝³ : CommRing R
        x✝² : CommRing S
        x✝¹ : CommRing T
        f : RingHom R S
        e : RingEquiv S T
        x✝ : (fun {R S} [CommRing R] [CommRing S] => RingHom.Locally fun {R S} [CommRi …
        s : Set S
        hsone : Eq (Ideal.span s) Top.top
        hs : ∀ (t : S), Membership.mem s t → (fun {R S} [CommRing R] [CommRing S] => P …
        ⊢ Eq (Ideal.span (Set.image (⇑e) s)) Top.top
      -/
    · rw [← Ideal.map_span, hsone, Ideal.map_top]
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
        R S T : Type u
        x✝³ : CommRing R
        x✝² : CommRing S
        x✝¹ : CommRing T
        f : RingHom R S
        e : RingEquiv S T
        x✝ : (fun {R S} [CommRing R] [CommRing S] => RingHom.Locally fun {R S} [CommRi …
        s : Set S
        hsone : Eq (Ideal.span s) Top.top
        hs : ∀ (t : S), Membership.mem s t → (fun {R S} [CommRing R] [CommRing S] => P …
        ⊢ ∀ (t : T), Membership.mem (Set.image (⇑e) s) t → (fun {R S} [CommRing R] [Co …
      -/
    · rintro - ⟨a, ha, rfl⟩
      let e' : Localization.Away a ≃+* Localization.Away (e a) :=
        IsLocalization.ringEquivOfRingEquiv _ _ e (Submonoid.map_powers e a)
      have : (algebraMap T (Localization.Away (e a))).comp e.toRingHom =
          e'.toRingHom.comp (algebraMap S (Localization.Away a)) := by
        ext x
        simp [e']
      /-
        case refine_2.intro.intro
        P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
        R S T : Type u
        x✝³ : CommRing R
        x✝² : CommRing S
        x✝¹ : CommRing T
        f : RingHom R S
        e : RingEquiv S T
        x✝ : (fun {R S} [CommRing R] [CommRing S] => RingHom.Locally fun {R S} [CommRi …
        s : Set S
        hsone : Eq (Ideal.span s) Top.top
        hs : ∀ (t : S), Membership.mem s t → (fun {R S} [CommRing R] [CommRing S] => P …
        a : S
        ha : Membership.mem s a
        e' : RingEquiv (Localization.Away a) (Localization.Away (e a)) := IsLocalizati …
        this : Eq ((algebraMap T (Localization.Away (e a))).comp e.toRingHom) (e'.toRi …
        ⊢ P ((algebraMap T (Localization.Away (e a))).comp (e.toRingHom.comp f))
      -/
      rw [← RingHom.comp_assoc, this, RingHom.comp_assoc]
      /-
        case refine_2.intro.intro
        P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
        R S T : Type u
        x✝³ : CommRing R
        x✝² : CommRing S
        x✝¹ : CommRing T
        f : RingHom R S
        e : RingEquiv S T
        x✝ : (fun {R S} [CommRing R] [CommRing S] => RingHom.Locally fun {R S} [CommRi …
        s : Set S
        hsone : Eq (Ideal.span s) Top.top
        hs : ∀ (t : S), Membership.mem s t → (fun {R S} [CommRing R] [CommRing S] => P …
        a : S
        ha : Membership.mem s a
        e' : RingEquiv (Localization.Away a) (Localization.Away (e a)) := IsLocalizati …
        this : Eq ((algebraMap T (Localization.Away (e a))).comp e.toRingHom) (e'.toRi …
        ⊢ P (e'.toRingHom.comp ((algebraMap S (Localization.Away a)).comp f))
      -/
      apply hPi.left
      /-
        case refine_2.intro.intro.x
        P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
        R S T : Type u
        x✝³ : CommRing R
        x✝² : CommRing S
        x✝¹ : CommRing T
        f : RingHom R S
        e : RingEquiv S T
        x✝ : (fun {R S} [CommRing R] [CommRing S] => RingHom.Locally fun {R S} [CommRi …
        s : Set S
        hsone : Eq (Ideal.span s) Top.top
        hs : ∀ (t : S), Membership.mem s t → (fun {R S} [CommRing R] [CommRing S] => P …
        a : S
        ha : Membership.mem s a
        e' : RingEquiv (Localization.Away a) (Localization.Away (e a)) := IsLocalizati …
        this : Eq ((algebraMap T (Localization.Away (e a))).comp e.toRingHom) (e'.toRi …
        ⊢ P ((algebraMap S (Localization.Away a)).comp f)
      -/
      exact hs a ha
      /-
        🎉 no goals
      -/
  right {R S T} _ _ _ f e := fun ⟨s, hsone, hs⟩ ↦
    ⟨s, hsone, fun a ha ↦ (RingHom.comp_assoc _ _ _).symm ▸ hPi.right _ _ (hs a ha)⟩


/-- If `P` holds for localization away maps, then so does `Locally P`. -/
lemma locally_holdsForLocalizationAway (hPa : HoldsForLocalizationAway P) :
    HoldsForLocalizationAway (Locally P) := by
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hPa : RingHom.HoldsForLocalizationAway fun {R S} [CommRing R] [CommRing S] => P
    ⊢ RingHom.HoldsForLocalizationAway fun {R S} [CommRing R] [CommRing S] => Ring …
  -/
  introv R _
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hPa : RingHom.HoldsForLocalizationAway fun {R S} [CommRing R] [CommRing S] => P
    R S : Type u
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    r : R
    inst✝ : IsLocalization.Away r S
    ⊢ RingHom.Locally (fun {R S} [CommRing R] [CommRing S] => P) (algebraMap R S)
  -/
  use {1}
  /-
    case h
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hPa : RingHom.HoldsForLocalizationAway fun {R S} [CommRing R] [CommRing S] => P
    R S : Type u
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    r : R
    inst✝ : IsLocalization.Away r S
    ⊢ Exists fun x => ∀ (t : S), Membership.mem (Singleton.singleton 1) t → (fun { …
  -/
  simp only [Set.mem_singleton_iff, forall_eq, Ideal.span_singleton_one, exists_const]
  let e : S ≃ₐ[R] (Localization.Away (1 : S)) :=
    (IsLocalization.atUnits S (Submonoid.powers 1) (by simp)).restrictScalars R
  haveI : IsLocalization.Away r (Localization.Away (1 : S)) :=
    IsLocalization.isLocalization_of_algEquiv (Submonoid.powers r) e
  /-
    case h
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hPa : RingHom.HoldsForLocalizationAway fun {R S} [CommRing R] [CommRing S] => P
    R S : Type u
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    r : R
    inst✝ : IsLocalization.Away r S
    e : AlgEquiv R S (Localization.Away 1) := AlgEquiv.restrictScalars R (IsLocali …
    this : IsLocalization.Away r (Localization.Away 1)
    ⊢ P ((algebraMap S (Localization.Away 1)).comp (algebraMap R S))
  -/
  rw [← IsScalarTower.algebraMap_eq]
  /-
    case h
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hPa : RingHom.HoldsForLocalizationAway fun {R S} [CommRing R] [CommRing S] => P
    R S : Type u
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    r : R
    inst✝ : IsLocalization.Away r S
    e : AlgEquiv R S (Localization.Away 1) := AlgEquiv.restrictScalars R (IsLocali …
    this : IsLocalization.Away r (Localization.Away 1)
    ⊢ P (algebraMap R (Localization.Away 1))
  -/
  apply hPa _ r
  /-
    🎉 no goals
  -/


/-- If `P` preserves localizations, then `Locally P` is stable under composition if `P` is. -/
lemma locally_stableUnderComposition (hPi : RespectsIso P) (hPl : LocalizationPreserves P)
    (hPc : StableUnderComposition P) :
    StableUnderComposition (Locally P) := by
  classical
  intro R S T _ _ _ f g hf hg
  rw [locally_iff_finite] at hf hg
  obtain ⟨sf, hsfone, hsf⟩ := hf
  obtain ⟨sg, hsgone, hsg⟩ := hg
  rw [locally_iff_exists hPi]
  refine ⟨sf × sg, fun (a, b) ↦ g a * b, ?_,
      fun (a, b) ↦ Localization.Away ((algebraMap T (Localization.Away b.val)) (g a.val)),
      inferInstance, inferInstance, inferInstance, ?_⟩
  · rw [eq_top_iff, ← hsgone, Ideal.span_le]
    intro t ht
    have : 1 ∈ Ideal.span (Set.range <| fun a : sf ↦ a.val) := by simp [hsfone]
    simp only [mem_ideal_span_range_iff_exists_fun, SetLike.mem_coe] at this ⊢
    obtain ⟨cf, hcf⟩ := this
    let cg : sg → T := Pi.single ⟨t, ht⟩ 1
    use fun (a, b) ↦ g (cf a) * cg b
    simp [cg, Pi.single_apply, Fintype.sum_prod_type, ← mul_assoc, ← Finset.sum_mul, ← map_mul,
      ← map_sum, hcf] at hcf ⊢
  · intro ⟨a, b⟩
    let g' := (algebraMap T (Localization.Away b.val)).comp g
    let a' := (algebraMap T (Localization.Away b.val)) (g a.val)
    have : (algebraMap T <| Localization.Away a').comp (g.comp f) =
        (Localization.awayMap g' a.val).comp ((algebraMap S (Localization.Away a.val)).comp f) := by
      ext x
      simp only [coe_comp, Function.comp_apply, a']
      change _ = Localization.awayMap g' a.val (algebraMap S _ (f x))
      simp only [Localization.awayMap, IsLocalization.Away.map, IsLocalization.map_eq]
      rfl
    simp only [this, a']
    apply hPc _ _ (hsf a.val a.property)
    apply @hPl _ _ _ _ g' _ _ _ _ _ _ _ _ ?_ (hsg b.val b.property)
    exact IsLocalization.Away.instMapRingHomPowersOfCoe (Localization.Away (g' a.val)) a.val


/-- If `P` is stable under composition with localization away maps on the right,
then so is `Locally P`. -/
lemma locally_StableUnderCompositionWithLocalizationAwayTarget
    (hP0 : RespectsIso P)
    (hPa : StableUnderCompositionWithLocalizationAwayTarget P) :
    StableUnderCompositionWithLocalizationAwayTarget (Locally P) := by
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP0 : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
    hPa : RingHom.StableUnderCompositionWithLocalizationAwayTarget fun {R S} [Comm …
    ⊢ RingHom.StableUnderCompositionWithLocalizationAwayTarget fun {R S} [CommRing …
  -/
  intro R S T _ _ _ _ t _ f hf
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP0 : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
    hPa : RingHom.StableUnderCompositionWithLocalizationAwayTarget fun {R S} [Comm …
    R S T : Type u
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : CommRing T
    inst✝¹ : Algebra S T
    t : S
    inst✝ : IsLocalization.Away t T
    f : RingHom R S
    hf : RingHom.Locally (fun {R S} [CommRing R] [CommRing S] => P) f
    ⊢ RingHom.Locally (fun {R S} [CommRing R] [CommRing S] => P) ((algebraMap S T) …
  -/
  simp only [locally_iff_isLocalization hP0 f] at hf
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP0 : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
    hPa : RingHom.StableUnderCompositionWithLocalizationAwayTarget fun {R S} [Comm …
    R S T : Type u
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : CommRing T
    inst✝¹ : Algebra S T
    t : S
    inst✝ : IsLocalization.Away t T
    f : RingHom R S
    hf : Exists fun s => Exists fun x => ∀ (t : S), Membership.mem s t → ∀ (Sₜ : T …
    ⊢ RingHom.Locally (fun {R S} [CommRing R] [CommRing S] => P) ((algebraMap S T) …
  -/
  obtain ⟨s, hsone, hs⟩ := hf
  /-
    case intro.intro
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP0 : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
    hPa : RingHom.StableUnderCompositionWithLocalizationAwayTarget fun {R S} [Comm …
    R S T : Type u
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : CommRing T
    inst✝¹ : Algebra S T
    t : S
    inst✝ : IsLocalization.Away t T
    f : RingHom R S
    s : Finset S
    hsone : Eq (Ideal.span ↑s) Top.top
    hs : ∀ (t : S), Membership.mem s t → ∀ (Sₜ : Type u) [inst : CommRing Sₜ] [ins …
    ⊢ RingHom.Locally (fun {R S} [CommRing R] [CommRing S] => P) ((algebraMap S T) …
  -/
  refine ⟨algebraMap S T '' s, ?_, ?_⟩
    /-
      case intro.intro.refine_1
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP0 : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      hPa : RingHom.StableUnderCompositionWithLocalizationAwayTarget fun {R S} [Comm …
      R S T : Type u
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : CommRing T
      inst✝¹ : Algebra S T
      t : S
      inst✝ : IsLocalization.Away t T
      f : RingHom R S
      s : Finset S
      hsone : Eq (Ideal.span ↑s) Top.top
      hs : ∀ (t : S), Membership.mem s t → ∀ (Sₜ : Type u) [inst : CommRing Sₜ] [ins …
      ⊢ Eq (Ideal.span (Set.image ⇑(algebraMap S T) ↑s)) Top.top
    -/
  · rw [← Ideal.map_span, hsone, Ideal.map_top]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP0 : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      hPa : RingHom.StableUnderCompositionWithLocalizationAwayTarget fun {R S} [Comm …
      R S T : Type u
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : CommRing T
      inst✝¹ : Algebra S T
      t : S
      inst✝ : IsLocalization.Away t T
      f : RingHom R S
      s : Finset S
      hsone : Eq (Ideal.span ↑s) Top.top
      hs : ∀ (t : S), Membership.mem s t → ∀ (Sₜ : Type u) [inst : CommRing Sₜ] [ins …
      ⊢ ∀ (t : T), Membership.mem (Set.image ⇑(algebraMap S T) ↑s) t → (fun {R S} [C …
    -/
  · rintro - ⟨a, ha, rfl⟩
    letI : Algebra (Localization.Away a) (Localization.Away (algebraMap S T a)) :=
      (IsLocalization.Away.map _ _ (algebraMap S T) a).toAlgebra
    have : (algebraMap (Localization.Away a) (Localization.Away (algebraMap S T a))).comp
        (algebraMap S (Localization.Away a)) =
        (algebraMap T (Localization.Away (algebraMap S T a))).comp (algebraMap S T) := by
      simp [algebraMap_toAlgebra, IsLocalization.Away.map]
    /-
      case intro.intro.refine_2.intro.intro
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP0 : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      hPa : RingHom.StableUnderCompositionWithLocalizationAwayTarget fun {R S} [Comm …
      R S T : Type u
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : CommRing T
      inst✝¹ : Algebra S T
      t : S
      inst✝ : IsLocalization.Away t T
      f : RingHom R S
      s : Finset S
      hsone : Eq (Ideal.span ↑s) Top.top
      hs : ∀ (t : S), Membership.mem s t → ∀ (Sₜ : Type u) [inst : CommRing Sₜ] [ins …
      a : S
      ha : Membership.mem (↑s) a
      this✝ : Algebra (Localization.Away a) (Localization.Away ((algebraMap S T) a)) …
      this : Eq ((algebraMap (Localization.Away a) (Localization.Away ((algebraMap S …
      ⊢ P ((algebraMap T (Localization.Away ((algebraMap S T) a))).comp ((algebraMap …
    -/
    rw [← comp_assoc, ← this, comp_assoc]
    haveI : IsScalarTower S (Localization.Away a) (Localization.Away ((algebraMap S T) a)) := by
      apply IsScalarTower.of_algebraMap_eq
      intro x
      simp [algebraMap_toAlgebra, IsLocalization.Away.map, ← IsScalarTower.algebraMap_apply]
    haveI : IsLocalization.Away (algebraMap S (Localization.Away a) t)
        (Localization.Away (algebraMap S T a)) :=
      IsLocalization.Away.commutes _ T ((Localization.Away (algebraMap S T a))) a t
    /-
      case intro.intro.refine_2.intro.intro
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP0 : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      hPa : RingHom.StableUnderCompositionWithLocalizationAwayTarget fun {R S} [Comm …
      R S T : Type u
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : CommRing T
      inst✝¹ : Algebra S T
      t : S
      inst✝ : IsLocalization.Away t T
      f : RingHom R S
      s : Finset S
      hsone : Eq (Ideal.span ↑s) Top.top
      hs : ∀ (t : S), Membership.mem s t → ∀ (Sₜ : Type u) [inst : CommRing Sₜ] [ins …
      a : S
      ha : Membership.mem (↑s) a
      this✝² : Algebra (Localization.Away a) (Localization.Away ((algebraMap S T) a) …
      this✝¹ : Eq ((algebraMap (Localization.Away a) (Localization.Away ((algebraMap …
      this✝ : IsScalarTower S (Localization.Away a) (Localization.Away ((algebraMap  …
      this : IsLocalization.Away ((algebraMap S (Localization.Away a)) t) (Localizat …
      ⊢ P ((algebraMap (Localization.Away a) (Localization.Away ((algebraMap S T) a) …
    -/
    apply hPa _ (algebraMap S (Localization.Away a) t)
    /-
      case intro.intro.refine_2.intro.intro.a
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP0 : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      hPa : RingHom.StableUnderCompositionWithLocalizationAwayTarget fun {R S} [Comm …
      R S T : Type u
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : CommRing T
      inst✝¹ : Algebra S T
      t : S
      inst✝ : IsLocalization.Away t T
      f : RingHom R S
      s : Finset S
      hsone : Eq (Ideal.span ↑s) Top.top
      hs : ∀ (t : S), Membership.mem s t → ∀ (Sₜ : Type u) [inst : CommRing Sₜ] [ins …
      a : S
      ha : Membership.mem (↑s) a
      this✝² : Algebra (Localization.Away a) (Localization.Away ((algebraMap S T) a) …
      this✝¹ : Eq ((algebraMap (Localization.Away a) (Localization.Away ((algebraMap …
      this✝ : IsScalarTower S (Localization.Away a) (Localization.Away ((algebraMap  …
      this : IsLocalization.Away ((algebraMap S (Localization.Away a)) t) (Localizat …
      ⊢ P ((algebraMap S (Localization.Away a)).comp f)
    -/
    apply hs a ha
    /-
      🎉 no goals
    -/


/-- If `P` is stable under composition with localization away maps on the left,
then so is `Locally P`. -/
lemma locally_StableUnderCompositionWithLocalizationAwaySource
    (hPa : StableUnderCompositionWithLocalizationAwaySource P) :
    StableUnderCompositionWithLocalizationAwaySource (Locally P) := by
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hPa : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [Comm …
    ⊢ RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommRing …
  -/
  intro R S T _ _ _ _ r _ f ⟨s, hsone, hs⟩
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hPa : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [Comm …
    R S T : Type u
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : CommRing T
    inst✝¹ : Algebra R S
    r : R
    inst✝ : IsLocalization.Away r S
    f : RingHom S T
    s : Set T
    hsone : Eq (Ideal.span s) Top.top
    hs : ∀ (t : T), Membership.mem s t → (fun {R S} [CommRing R] [CommRing S] => P …
    ⊢ RingHom.Locally (fun {R S} [CommRing R] [CommRing S] => P) (f.comp (algebraM …
  -/
  refine ⟨s, hsone, fun t ht ↦ ?_⟩
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hPa : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [Comm …
    R S T : Type u
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : CommRing T
    inst✝¹ : Algebra R S
    r : R
    inst✝ : IsLocalization.Away r S
    f : RingHom S T
    s : Set T
    hsone : Eq (Ideal.span s) Top.top
    hs : ∀ (t : T), Membership.mem s t → (fun {R S} [CommRing R] [CommRing S] => P …
    t : T
    ht : Membership.mem s t
    ⊢ (fun {R S} [CommRing R] [CommRing S] => P) ((algebraMap T (Localization.Away …
  -/
  rw [← comp_assoc]
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hPa : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [Comm …
    R S T : Type u
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : CommRing T
    inst✝¹ : Algebra R S
    r : R
    inst✝ : IsLocalization.Away r S
    f : RingHom S T
    s : Set T
    hsone : Eq (Ideal.span s) Top.top
    hs : ∀ (t : T), Membership.mem s t → (fun {R S} [CommRing R] [CommRing S] => P …
    t : T
    ht : Membership.mem s t
    ⊢ (fun {R S} [CommRing R] [CommRing S] => P) (((algebraMap T (Localization.Awa …
  -/
  exact hPa _ r _ (hs t ht)
  /-
    🎉 no goals
  -/


attribute [local instance] Algebra.TensorProduct.rightAlgebra in
/-- If `P` is stable under base change, then so is `Locally P`. -/
lemma locally_isStableUnderBaseChange (hPi : RespectsIso P) (hPb : IsStableUnderBaseChange P) :
    IsStableUnderBaseChange (Locally P) := by
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
    hPb : RingHom.IsStableUnderBaseChange fun {R S} [CommRing R] [CommRing S] => P
    ⊢ RingHom.IsStableUnderBaseChange fun {R S} [CommRing R] [CommRing S] => RingH …
  -/
  apply IsStableUnderBaseChange.mk _ (locally_respectsIso hPi)
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
    hPb : RingHom.IsStableUnderBaseChange fun {R S} [CommRing R] [CommRing S] => P
    ⊢ ∀ ⦃R S T : Type u⦄ [inst : CommRing R] [inst_1 : CommRing S] [inst_2 : CommR …
  -/
  introv hf
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
    hPb : RingHom.IsStableUnderBaseChange fun {R S} [CommRing R] [CommRing S] => P
    R S T : Type u
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : CommRing T
    inst✝¹ : Algebra R S
    inst✝ : Algebra R T
    hf : RingHom.Locally (fun {R S} [CommRing R] [CommRing S] => P) (algebraMap R T)
    ⊢ RingHom.Locally (fun {R S} [CommRing R] [CommRing S] => P) Algebra.TensorPro …
  -/
  obtain ⟨s, hsone, hs⟩ := hf
  /-
    case intro.intro
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
    hPb : RingHom.IsStableUnderBaseChange fun {R S} [CommRing R] [CommRing S] => P
    R S T : Type u
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : CommRing T
    inst✝¹ : Algebra R S
    inst✝ : Algebra R T
    s : Set T
    hsone : Eq (Ideal.span s) Top.top
    hs : ∀ (t : T), Membership.mem s t → (fun {R S} [CommRing R] [CommRing S] => P …
    ⊢ RingHom.Locally (fun {R S} [CommRing R] [CommRing S] => P) Algebra.TensorPro …
  -/
  rw [locally_iff_exists hPi]
  letI (a : s) : Algebra (S ⊗[R] T) (S ⊗[R] Localization.Away a.val) :=
    (Algebra.TensorProduct.map (AlgHom.id R S) (IsScalarTower.toAlgHom R _ _)).toRingHom.toAlgebra
  letI (a : s) : Algebra T (S ⊗[R] Localization.Away a.val) :=
    ((algebraMap _ (S ⊗[R] Localization.Away a.val)).comp (algebraMap T (S ⊗[R] T))).toAlgebra
  haveI (a : s) : IsScalarTower T (S ⊗[R] T) (S ⊗[R] Localization.Away a.val) :=
    IsScalarTower.of_algebraMap_eq' rfl
  haveI (a : s) : IsScalarTower T (Localization.Away a.val) (S ⊗[R] Localization.Away a.val) :=
    IsScalarTower.of_algebraMap_eq' rfl
  haveI (a : s) : IsScalarTower S (S ⊗[R] T) (S ⊗[R] Localization.Away a.val) :=
      IsScalarTower.of_algebraMap_eq <| by
    intro x
    simp [RingHom.algebraMap_toAlgebra]
  haveI (a : s) : Algebra.IsPushout T (Localization.Away a.val) (S ⊗[R] T)
      (S ⊗[R] Localization.Away a.val) := by
    rw [← Algebra.IsPushout.comp_iff (R := R) (R' := S)]
    infer_instance
  refine ⟨s, fun a ↦ Algebra.TensorProduct.includeRight a.val, ?_,
      fun a ↦ (S ⊗[R] Localization.Away a.val), inferInstance, inferInstance, ?_, ?_⟩
    /-
      case intro.intro.refine_1
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      hPb : RingHom.IsStableUnderBaseChange fun {R S} [CommRing R] [CommRing S] => P
      R S T : Type u
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : CommRing T
      inst✝¹ : Algebra R S
      inst✝ : Algebra R T
      s : Set T
      hsone : Eq (Ideal.span s) Top.top
      hs : ∀ (t : T), Membership.mem s t → (fun {R S} [CommRing R] [CommRing S] => P …
      this✝⁴ : (a : ↑s) → Algebra (TensorProduct R S T) (TensorProduct R S (Localiza …
      this✝³ : (a : ↑s) → Algebra T (TensorProduct R S (Localization.Away ↑a)) := fu …
      this✝² : ∀ (a : ↑s), IsScalarTower T (TensorProduct R S T) (TensorProduct R S  …
      this✝¹ : ∀ (a : ↑s), IsScalarTower T (Localization.Away ↑a) (TensorProduct R S …
      this✝ : ∀ (a : ↑s), IsScalarTower S (TensorProduct R S T) (TensorProduct R S ( …
      this : ∀ (a : ↑s), Algebra.IsPushout T (Localization.Away ↑a) (TensorProduct R …
      ⊢ Eq (Ideal.span (Set.range fun a => Algebra.TensorProduct.includeRight ↑a)) T …
    -/
  · rw [← Set.image_eq_range, ← Ideal.map_span, hsone, Ideal.map_top]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      hPb : RingHom.IsStableUnderBaseChange fun {R S} [CommRing R] [CommRing S] => P
      R S T : Type u
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : CommRing T
      inst✝¹ : Algebra R S
      inst✝ : Algebra R T
      s : Set T
      hsone : Eq (Ideal.span s) Top.top
      hs : ∀ (t : T), Membership.mem s t → (fun {R S} [CommRing R] [CommRing S] => P …
      this✝⁴ : (a : ↑s) → Algebra (TensorProduct R S T) (TensorProduct R S (Localiza …
      this✝³ : (a : ↑s) → Algebra T (TensorProduct R S (Localization.Away ↑a)) := fu …
      this✝² : ∀ (a : ↑s), IsScalarTower T (TensorProduct R S T) (TensorProduct R S  …
      this✝¹ : ∀ (a : ↑s), IsScalarTower T (Localization.Away ↑a) (TensorProduct R S …
      this✝ : ∀ (a : ↑s), IsScalarTower S (TensorProduct R S T) (TensorProduct R S ( …
      this : ∀ (a : ↑s), Algebra.IsPushout T (Localization.Away ↑a) (TensorProduct R …
      ⊢ ∀ (i : ↑s), IsLocalization.Away ((fun a => Algebra.TensorProduct.includeRigh …
    -/
  · intro a
    convert_to IsLocalization (Algebra.algebraMapSubmonoid (S ⊗[R] T) (Submonoid.powers a.val))
        (S ⊗[R] Localization.Away a.val)
    · simp only [Algebra.TensorProduct.includeRight_apply, Algebra.algebraMapSubmonoid,
        Submonoid.map_powers]
      /-
        case h.e
        P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
        hPb : RingHom.IsStableUnderBaseChange fun {R S} [CommRing R] [CommRing S] => P
        R S T : Type u
        inst✝⁴ : CommRing R
        inst✝³ : CommRing S
        inst✝² : CommRing T
        inst✝¹ : Algebra R S
        inst✝ : Algebra R T
        s : Set T
        hsone : Eq (Ideal.span s) Top.top
        hs : ∀ (t : T), Membership.mem s t → (fun {R S} [CommRing R] [CommRing S] => P …
        this✝⁴ : (a : ↑s) → Algebra (TensorProduct R S T) (TensorProduct R S (Localiza …
        this✝³ : (a : ↑s) → Algebra T (TensorProduct R S (Localization.Away ↑a)) := fu …
        this✝² : ∀ (a : ↑s), IsScalarTower T (TensorProduct R S T) (TensorProduct R S  …
        this✝¹ : ∀ (a : ↑s), IsScalarTower T (Localization.Away ↑a) (TensorProduct R S …
        this✝ : ∀ (a : ↑s), IsScalarTower S (TensorProduct R S T) (TensorProduct R S ( …
        this : ∀ (a : ↑s), Algebra.IsPushout T (Localization.Away ↑a) (TensorProduct R …
        a : ↑s
        ⊢ Eq (IsLocalization.Away (TensorProduct.tmul R 1 ↑a)) (IsLocalization (Submon …
      -/
      rfl
      /-
        🎉 no goals
      -/
    · rw [← isLocalizedModule_iff_isLocalization, isLocalizedModule_iff_isBaseChange
        (S := Submonoid.powers a.val) (A := Localization.Away a.val)]
      /-
        case intro.intro.refine_2
        P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
        hPb : RingHom.IsStableUnderBaseChange fun {R S} [CommRing R] [CommRing S] => P
        R S T : Type u
        inst✝⁴ : CommRing R
        inst✝³ : CommRing S
        inst✝² : CommRing T
        inst✝¹ : Algebra R S
        inst✝ : Algebra R T
        s : Set T
        hsone : Eq (Ideal.span s) Top.top
        hs : ∀ (t : T), Membership.mem s t → (fun {R S} [CommRing R] [CommRing S] => P …
        this✝⁴ : (a : ↑s) → Algebra (TensorProduct R S T) (TensorProduct R S (Localiza …
        this✝³ : (a : ↑s) → Algebra T (TensorProduct R S (Localization.Away ↑a)) := fu …
        this✝² : ∀ (a : ↑s), IsScalarTower T (TensorProduct R S T) (TensorProduct R S  …
        this✝¹ : ∀ (a : ↑s), IsScalarTower T (Localization.Away ↑a) (TensorProduct R S …
        this✝ : ∀ (a : ↑s), IsScalarTower S (TensorProduct R S T) (TensorProduct R S ( …
        this : ∀ (a : ↑s), Algebra.IsPushout T (Localization.Away ↑a) (TensorProduct R …
        a : ↑s
        ⊢ IsBaseChange (Localization.Away ↑a) (IsScalarTower.toAlgHom T (TensorProduct …
      -/
      exact Algebra.IsPushout.out
      /-
        🎉 no goals
      -/
    /-
      case intro.intro.refine_3
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      hPb : RingHom.IsStableUnderBaseChange fun {R S} [CommRing R] [CommRing S] => P
      R S T : Type u
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : CommRing T
      inst✝¹ : Algebra R S
      inst✝ : Algebra R T
      s : Set T
      hsone : Eq (Ideal.span s) Top.top
      hs : ∀ (t : T), Membership.mem s t → (fun {R S} [CommRing R] [CommRing S] => P …
      this✝⁴ : (a : ↑s) → Algebra (TensorProduct R S T) (TensorProduct R S (Localiza …
      this✝³ : (a : ↑s) → Algebra T (TensorProduct R S (Localization.Away ↑a)) := fu …
      this✝² : ∀ (a : ↑s), IsScalarTower T (TensorProduct R S T) (TensorProduct R S  …
      this✝¹ : ∀ (a : ↑s), IsScalarTower T (Localization.Away ↑a) (TensorProduct R S …
      this✝ : ∀ (a : ↑s), IsScalarTower S (TensorProduct R S T) (TensorProduct R S ( …
      this : ∀ (a : ↑s), Algebra.IsPushout T (Localization.Away ↑a) (TensorProduct R …
      ⊢ ∀ (i : ↑s), P ((algebraMap (TensorProduct R S T) ((fun a => TensorProduct R  …
    -/
  · intro a
    have : (algebraMap (S ⊗[R] T) (S ⊗[R] Localization.Away a.val)).comp
        Algebra.TensorProduct.includeLeftRingHom =
        Algebra.TensorProduct.includeLeftRingHom := by
      ext x
      simp [RingHom.algebraMap_toAlgebra]
    /-
      case intro.intro.refine_3
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      hPb : RingHom.IsStableUnderBaseChange fun {R S} [CommRing R] [CommRing S] => P
      R S T : Type u
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : CommRing T
      inst✝¹ : Algebra R S
      inst✝ : Algebra R T
      s : Set T
      hsone : Eq (Ideal.span s) Top.top
      hs : ∀ (t : T), Membership.mem s t → (fun {R S} [CommRing R] [CommRing S] => P …
      this✝⁵ : (a : ↑s) → Algebra (TensorProduct R S T) (TensorProduct R S (Localiza …
      this✝⁴ : (a : ↑s) → Algebra T (TensorProduct R S (Localization.Away ↑a)) := fu …
      this✝³ : ∀ (a : ↑s), IsScalarTower T (TensorProduct R S T) (TensorProduct R S  …
      this✝² : ∀ (a : ↑s), IsScalarTower T (Localization.Away ↑a) (TensorProduct R S …
      this✝¹ : ∀ (a : ↑s), IsScalarTower S (TensorProduct R S T) (TensorProduct R S  …
      this✝ : ∀ (a : ↑s), Algebra.IsPushout T (Localization.Away ↑a) (TensorProduct  …
      a : ↑s
      this : Eq ((algebraMap (TensorProduct R S T) (TensorProduct R S (Localization. …
      ⊢ P ((algebraMap (TensorProduct R S T) ((fun a => TensorProduct R S (Localizat …
    -/
    rw [this]
    /-
      case intro.intro.refine_3
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      hPb : RingHom.IsStableUnderBaseChange fun {R S} [CommRing R] [CommRing S] => P
      R S T : Type u
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : CommRing T
      inst✝¹ : Algebra R S
      inst✝ : Algebra R T
      s : Set T
      hsone : Eq (Ideal.span s) Top.top
      hs : ∀ (t : T), Membership.mem s t → (fun {R S} [CommRing R] [CommRing S] => P …
      this✝⁵ : (a : ↑s) → Algebra (TensorProduct R S T) (TensorProduct R S (Localiza …
      this✝⁴ : (a : ↑s) → Algebra T (TensorProduct R S (Localization.Away ↑a)) := fu …
      this✝³ : ∀ (a : ↑s), IsScalarTower T (TensorProduct R S T) (TensorProduct R S  …
      this✝² : ∀ (a : ↑s), IsScalarTower T (Localization.Away ↑a) (TensorProduct R S …
      this✝¹ : ∀ (a : ↑s), IsScalarTower S (TensorProduct R S T) (TensorProduct R S  …
      this✝ : ∀ (a : ↑s), Algebra.IsPushout T (Localization.Away ↑a) (TensorProduct  …
      a : ↑s
      this : Eq ((algebraMap (TensorProduct R S T) (TensorProduct R S (Localization. …
      ⊢ P Algebra.TensorProduct.includeLeftRingHom
    -/
    apply hPb R (Localization.Away a.val)
    /-
      case intro.intro.refine_3.a
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      hPb : RingHom.IsStableUnderBaseChange fun {R S} [CommRing R] [CommRing S] => P
      R S T : Type u
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : CommRing T
      inst✝¹ : Algebra R S
      inst✝ : Algebra R T
      s : Set T
      hsone : Eq (Ideal.span s) Top.top
      hs : ∀ (t : T), Membership.mem s t → (fun {R S} [CommRing R] [CommRing S] => P …
      this✝⁵ : (a : ↑s) → Algebra (TensorProduct R S T) (TensorProduct R S (Localiza …
      this✝⁴ : (a : ↑s) → Algebra T (TensorProduct R S (Localization.Away ↑a)) := fu …
      this✝³ : ∀ (a : ↑s), IsScalarTower T (TensorProduct R S T) (TensorProduct R S  …
      this✝² : ∀ (a : ↑s), IsScalarTower T (Localization.Away ↑a) (TensorProduct R S …
      this✝¹ : ∀ (a : ↑s), IsScalarTower S (TensorProduct R S T) (TensorProduct R S  …
      this✝ : ∀ (a : ↑s), Algebra.IsPushout T (Localization.Away ↑a) (TensorProduct  …
      a : ↑s
      this : Eq ((algebraMap (TensorProduct R S T) (TensorProduct R S (Localization. …
      ⊢ P (algebraMap R (Localization.Away ↑a))
    -/
    rw [IsScalarTower.algebraMap_eq R T (Localization.Away a.val)]
    /-
      case intro.intro.refine_3.a
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      hPb : RingHom.IsStableUnderBaseChange fun {R S} [CommRing R] [CommRing S] => P
      R S T : Type u
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : CommRing T
      inst✝¹ : Algebra R S
      inst✝ : Algebra R T
      s : Set T
      hsone : Eq (Ideal.span s) Top.top
      hs : ∀ (t : T), Membership.mem s t → (fun {R S} [CommRing R] [CommRing S] => P …
      this✝⁵ : (a : ↑s) → Algebra (TensorProduct R S T) (TensorProduct R S (Localiza …
      this✝⁴ : (a : ↑s) → Algebra T (TensorProduct R S (Localization.Away ↑a)) := fu …
      this✝³ : ∀ (a : ↑s), IsScalarTower T (TensorProduct R S T) (TensorProduct R S  …
      this✝² : ∀ (a : ↑s), IsScalarTower T (Localization.Away ↑a) (TensorProduct R S …
      this✝¹ : ∀ (a : ↑s), IsScalarTower S (TensorProduct R S T) (TensorProduct R S  …
      this✝ : ∀ (a : ↑s), Algebra.IsPushout T (Localization.Away ↑a) (TensorProduct  …
      a : ↑s
      this : Eq ((algebraMap (TensorProduct R S T) (TensorProduct R S (Localization. …
      ⊢ P ((algebraMap T (Localization.Away ↑a)).comp (algebraMap R T))
    -/
    apply hs a a.property
    /-
      🎉 no goals
    -/


/-- If `P` is preserved by localization away, then so is `Locally P`. -/
lemma locally_localizationAwayPreserves (hPl : LocalizationAwayPreserves P) :
    LocalizationAwayPreserves (Locally P) := by
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hPl : RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => P
    ⊢ RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => Rin …
  -/
  introv R hf
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hPl : RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => P
    R S : Type u
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    f : RingHom R S
    r : R
    R' S' : Type u
    inst✝⁵ : CommRing R'
    inst✝⁴ : CommRing S'
    inst✝³ : Algebra R R'
    inst✝² : Algebra S S'
    inst✝¹ : IsLocalization.Away r R'
    inst✝ : IsLocalization.Away (f r) S'
    hf : RingHom.Locally (fun {R S} [CommRing R] [CommRing S] => P) f
    ⊢ RingHom.Locally (fun {R S} [CommRing R] [CommRing S] => P) (IsLocalization.A …
  -/
  obtain ⟨s, hsone, hs⟩ := hf
  /-
    case intro.intro
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hPl : RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => P
    R S : Type u
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    f : RingHom R S
    r : R
    R' S' : Type u
    inst✝⁵ : CommRing R'
    inst✝⁴ : CommRing S'
    inst✝³ : Algebra R R'
    inst✝² : Algebra S S'
    inst✝¹ : IsLocalization.Away r R'
    inst✝ : IsLocalization.Away (f r) S'
    s : Set S
    hsone : Eq (Ideal.span s) Top.top
    hs : ∀ (t : S), Membership.mem s t → (fun {R S} [CommRing R] [CommRing S] => P …
    ⊢ RingHom.Locally (fun {R S} [CommRing R] [CommRing S] => P) (IsLocalization.A …
  -/
  rw [locally_iff_exists hPl.respectsIso]
  /-
    case intro.intro
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hPl : RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => P
    R S : Type u
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    f : RingHom R S
    r : R
    R' S' : Type u
    inst✝⁵ : CommRing R'
    inst✝⁴ : CommRing S'
    inst✝³ : Algebra R R'
    inst✝² : Algebra S S'
    inst✝¹ : IsLocalization.Away r R'
    inst✝ : IsLocalization.Away (f r) S'
    s : Set S
    hsone : Eq (Ideal.span s) Top.top
    hs : ∀ (t : S), Membership.mem s t → (fun {R S} [CommRing R] [CommRing S] => P …
    ⊢ Exists fun ι => Exists fun s => Exists fun x => Exists fun Sₜ => Exists fun  …
  -/
  let rₐ (a : s) : Localization.Away a.val := algebraMap _ _ (f r)
  /-
    case intro.intro
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hPl : RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => P
    R S : Type u
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    f : RingHom R S
    r : R
    R' S' : Type u
    inst✝⁵ : CommRing R'
    inst✝⁴ : CommRing S'
    inst✝³ : Algebra R R'
    inst✝² : Algebra S S'
    inst✝¹ : IsLocalization.Away r R'
    inst✝ : IsLocalization.Away (f r) S'
    s : Set S
    hsone : Eq (Ideal.span s) Top.top
    hs : ∀ (t : S), Membership.mem s t → (fun {R S} [CommRing R] [CommRing S] => P …
    rₐ : (a : ↑s) → Localization.Away ↑a := fun a => (algebraMap S (Localization.A …
    ⊢ Exists fun ι => Exists fun s => Exists fun x => Exists fun Sₜ => Exists fun  …
  -/
  let Sₐ (a : s) := Localization.Away (rₐ a)
  haveI (a : s) :
      IsLocalization.Away (((algebraMap S (Localization.Away a.val)).comp f) r) (Sₐ a) :=
    inferInstanceAs (IsLocalization.Away (rₐ a) (Sₐ a))
  haveI (a : s) : IsLocalization (Algebra.algebraMapSubmonoid (Localization.Away a.val)
    (Submonoid.map f (Submonoid.powers r))) (Sₐ a) := by
    convert inferInstanceAs (IsLocalization.Away (rₐ a) (Sₐ a))
    simp [rₐ, Sₐ, Algebra.algebraMapSubmonoid]
  have H (a : s) : Submonoid.powers (f r) ≤
      (Submonoid.powers (rₐ a)).comap (algebraMap S (Localization.Away a.val)) := by
    simp [rₐ, Sₐ, Submonoid.powers_le]
  letI (a : s) : Algebra S' (Sₐ a) :=
    (IsLocalization.map (Sₐ a) (algebraMap S (Localization.Away a.val)) (H a)).toAlgebra
  haveI (a : s) : IsScalarTower S S' (Sₐ a) :=
    IsScalarTower.of_algebraMap_eq' (IsLocalization.map_comp (H a)).symm
  refine ⟨s, fun a ↦ algebraMap S S' a.val, ?_, Sₐ,
      inferInstance, inferInstance, fun a ↦ ?_, fun a ↦ ?_⟩
    /-
      case intro.intro.refine_1
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hPl : RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => P
      R S : Type u
      inst✝⁷ : CommRing R
      inst✝⁶ : CommRing S
      f : RingHom R S
      r : R
      R' S' : Type u
      inst✝⁵ : CommRing R'
      inst✝⁴ : CommRing S'
      inst✝³ : Algebra R R'
      inst✝² : Algebra S S'
      inst✝¹ : IsLocalization.Away r R'
      inst✝ : IsLocalization.Away (f r) S'
      s : Set S
      hsone : Eq (Ideal.span s) Top.top
      hs : ∀ (t : S), Membership.mem s t → (fun {R S} [CommRing R] [CommRing S] => P …
      rₐ : (a : ↑s) → Localization.Away ↑a := fun a => (algebraMap S (Localization.A …
      Sₐ : ↑s → Type u := fun a => Localization.Away (rₐ a)
      this✝² : ∀ (a : ↑s), IsLocalization.Away (((algebraMap S (Localization.Away ↑a …
      this✝¹ : ∀ (a : ↑s), IsLocalization (Algebra.algebraMapSubmonoid (Localization …
      H : ∀ (a : ↑s), LE.le (Submonoid.powers (f r)) (Submonoid.comap (algebraMap S  …
      this✝ : (a : ↑s) → Algebra S' (Sₐ a) := fun a => (IsLocalization.map (Sₐ a) (a …
      this : ∀ (a : ↑s), IsScalarTower S S' (Sₐ a)
      ⊢ Eq (Ideal.span (Set.range fun a => (algebraMap S S') ↑a)) Top.top
    -/
  · rw [← Set.image_eq_range, ← Ideal.map_span, hsone, Ideal.map_top]
    /-
      🎉 no goals
    -/
  · convert IsLocalization.commutes (T := Sₐ a) (M₁ := (Submonoid.powers r).map f) (S₁ := S')
      (S₂ := Localization.Away a.val) (M₂ := Submonoid.powers a.val)
    /-
      case h.e.h.e'_3.h
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hPl : RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => P
      R S : Type u
      inst✝⁷ : CommRing R
      inst✝⁶ : CommRing S
      f : RingHom R S
      r : R
      R' S' : Type u
      inst✝⁵ : CommRing R'
      inst✝⁴ : CommRing S'
      inst✝³ : Algebra R R'
      inst✝² : Algebra S S'
      inst✝¹ : IsLocalization.Away r R'
      inst✝ : IsLocalization.Away (f r) S'
      s : Set S
      hsone : Eq (Ideal.span s) Top.top
      hs : ∀ (t : S), Membership.mem s t → (fun {R S} [CommRing R] [CommRing S] => P …
      rₐ : (a : ↑s) → Localization.Away ↑a := fun a => (algebraMap S (Localization.A …
      Sₐ : ↑s → Type u := fun a => Localization.Away (rₐ a)
      this✝² : ∀ (a : ↑s), IsLocalization.Away (((algebraMap S (Localization.Away ↑a …
      this✝¹ : ∀ (a : ↑s), IsLocalization (Algebra.algebraMapSubmonoid (Localization …
      H : ∀ (a : ↑s), LE.le (Submonoid.powers (f r)) (Submonoid.comap (algebraMap S  …
      this✝ : (a : ↑s) → Algebra S' (Sₐ a) := fun a => (IsLocalization.map (Sₐ a) (a …
      this : ∀ (a : ↑s), IsScalarTower S S' (Sₐ a)
      a : ↑s
      ⊢ Eq (Submonoid.powers ((fun a => (algebraMap S S') ↑a) a)) (Algebra.algebraMa …
    -/
    simp [Algebra.algebraMapSubmonoid]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_3
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hPl : RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => P
      R S : Type u
      inst✝⁷ : CommRing R
      inst✝⁶ : CommRing S
      f : RingHom R S
      r : R
      R' S' : Type u
      inst✝⁵ : CommRing R'
      inst✝⁴ : CommRing S'
      inst✝³ : Algebra R R'
      inst✝² : Algebra S S'
      inst✝¹ : IsLocalization.Away r R'
      inst✝ : IsLocalization.Away (f r) S'
      s : Set S
      hsone : Eq (Ideal.span s) Top.top
      hs : ∀ (t : S), Membership.mem s t → (fun {R S} [CommRing R] [CommRing S] => P …
      rₐ : (a : ↑s) → Localization.Away ↑a := fun a => (algebraMap S (Localization.A …
      Sₐ : ↑s → Type u := fun a => Localization.Away (rₐ a)
      this✝² : ∀ (a : ↑s), IsLocalization.Away (((algebraMap S (Localization.Away ↑a …
      this✝¹ : ∀ (a : ↑s), IsLocalization (Algebra.algebraMapSubmonoid (Localization …
      H : ∀ (a : ↑s), LE.le (Submonoid.powers (f r)) (Submonoid.comap (algebraMap S  …
      this✝ : (a : ↑s) → Algebra S' (Sₐ a) := fun a => (IsLocalization.map (Sₐ a) (a …
      this : ∀ (a : ↑s), IsScalarTower S S' (Sₐ a)
      a : ↑s
      ⊢ P ((algebraMap S' (Sₐ a)).comp (IsLocalization.Away.map R' S' f r))
    -/
  · rw [algebraMap_toAlgebra, IsLocalization.Away.map, IsLocalization.map_comp_map]
    /-
      case intro.intro.refine_3
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hPl : RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => P
      R S : Type u
      inst✝⁷ : CommRing R
      inst✝⁶ : CommRing S
      f : RingHom R S
      r : R
      R' S' : Type u
      inst✝⁵ : CommRing R'
      inst✝⁴ : CommRing S'
      inst✝³ : Algebra R R'
      inst✝² : Algebra S S'
      inst✝¹ : IsLocalization.Away r R'
      inst✝ : IsLocalization.Away (f r) S'
      s : Set S
      hsone : Eq (Ideal.span s) Top.top
      hs : ∀ (t : S), Membership.mem s t → (fun {R S} [CommRing R] [CommRing S] => P …
      rₐ : (a : ↑s) → Localization.Away ↑a := fun a => (algebraMap S (Localization.A …
      Sₐ : ↑s → Type u := fun a => Localization.Away (rₐ a)
      this✝² : ∀ (a : ↑s), IsLocalization.Away (((algebraMap S (Localization.Away ↑a …
      this✝¹ : ∀ (a : ↑s), IsLocalization (Algebra.algebraMapSubmonoid (Localization …
      H : ∀ (a : ↑s), LE.le (Submonoid.powers (f r)) (Submonoid.comap (algebraMap S  …
      this✝ : (a : ↑s) → Algebra S' (Sₐ a) := fun a => (IsLocalization.map (Sₐ a) (a …
      this : ∀ (a : ↑s), IsScalarTower S S' (Sₐ a)
      a : ↑s
      ⊢ P (IsLocalization.map (Sₐ a) ((algebraMap S (Localization.Away ↑a)).comp f) ⋯)
    -/
    exact hPl ((algebraMap _ (Localization.Away a.val)).comp f) r R' (Sₐ a) (hs _ a.2)
    /-
      🎉 no goals
    -/


/-- If `P` is preserved by localizations, then so is `Locally P`. -/
lemma locally_localizationPreserves (hPl : LocalizationPreserves P) :
    LocalizationPreserves (Locally P) := by
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hPl : RingHom.LocalizationPreserves fun {R S} [CommRing R] [CommRing S] => P
    ⊢ RingHom.LocalizationPreserves fun {R S} [CommRing R] [CommRing S] => RingHom …
  -/
  introv R hf
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hPl : RingHom.LocalizationPreserves fun {R S} [CommRing R] [CommRing S] => P
    R S : Type u
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    f : RingHom R S
    M : Submonoid R
    R' S' : Type u
    inst✝⁵ : CommRing R'
    inst✝⁴ : CommRing S'
    inst✝³ : Algebra R R'
    inst✝² : Algebra S S'
    inst✝¹ : IsLocalization M R'
    inst✝ : IsLocalization (Submonoid.map f M) S'
    hf : RingHom.Locally (fun {R S} [CommRing R] [CommRing S] => P) f
    ⊢ RingHom.Locally (fun {R S} [CommRing R] [CommRing S] => P) (IsLocalization.m …
  -/
  obtain ⟨s, hsone, hs⟩ := hf
  /-
    case intro.intro
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hPl : RingHom.LocalizationPreserves fun {R S} [CommRing R] [CommRing S] => P
    R S : Type u
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    f : RingHom R S
    M : Submonoid R
    R' S' : Type u
    inst✝⁵ : CommRing R'
    inst✝⁴ : CommRing S'
    inst✝³ : Algebra R R'
    inst✝² : Algebra S S'
    inst✝¹ : IsLocalization M R'
    inst✝ : IsLocalization (Submonoid.map f M) S'
    s : Set S
    hsone : Eq (Ideal.span s) Top.top
    hs : ∀ (t : S), Membership.mem s t → (fun {R S} [CommRing R] [CommRing S] => P …
    ⊢ RingHom.Locally (fun {R S} [CommRing R] [CommRing S] => P) (IsLocalization.m …
  -/
  rw [locally_iff_exists hPl.away.respectsIso]
  let Mₐ (a : s) : Submonoid (Localization.Away a.val) :=
    (M.map f).map (algebraMap S (Localization.Away a.val))
  /-
    case intro.intro
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hPl : RingHom.LocalizationPreserves fun {R S} [CommRing R] [CommRing S] => P
    R S : Type u
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    f : RingHom R S
    M : Submonoid R
    R' S' : Type u
    inst✝⁵ : CommRing R'
    inst✝⁴ : CommRing S'
    inst✝³ : Algebra R R'
    inst✝² : Algebra S S'
    inst✝¹ : IsLocalization M R'
    inst✝ : IsLocalization (Submonoid.map f M) S'
    s : Set S
    hsone : Eq (Ideal.span s) Top.top
    hs : ∀ (t : S), Membership.mem s t → (fun {R S} [CommRing R] [CommRing S] => P …
    Mₐ : (a : ↑s) → Submonoid (Localization.Away ↑a) := fun a => Submonoid.map (al …
    ⊢ Exists fun ι => Exists fun s => Exists fun x => Exists fun Sₜ => Exists fun  …
  -/
  let Sₐ (a : s) := Localization (Mₐ a)
  have hM (a : s) : M.map ((algebraMap S (Localization.Away a.val)).comp f) = Mₐ a :=
    (M.map_map _ _).symm
  haveI (a : s) :
      IsLocalization (M.map ((algebraMap S (Localization.Away a.val)).comp f)) (Sₐ a) := by
    rw [hM]
    infer_instance
  haveI (a : s) :
      IsLocalization (Algebra.algebraMapSubmonoid (Localization.Away a.val) (M.map f)) (Sₐ a) :=
    inferInstanceAs <| IsLocalization (Mₐ a) (Sₐ a)
  letI (a : s) : Algebra S' (Sₐ a) :=
    (IsLocalization.map (Sₐ a) (algebraMap S (Localization.Away a.val))
      (M.map f).le_comap_map).toAlgebra
  haveI (a : s) : IsScalarTower S S' (Sₐ a) :=
    IsScalarTower.of_algebraMap_eq' (IsLocalization.map_comp (M.map f).le_comap_map).symm
  refine ⟨s, fun a ↦ algebraMap S S' a.val, ?_, Sₐ,
      inferInstance, inferInstance, fun a ↦ ?_, fun a ↦ ?_⟩
    /-
      case intro.intro.refine_1
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hPl : RingHom.LocalizationPreserves fun {R S} [CommRing R] [CommRing S] => P
      R S : Type u
      inst✝⁷ : CommRing R
      inst✝⁶ : CommRing S
      f : RingHom R S
      M : Submonoid R
      R' S' : Type u
      inst✝⁵ : CommRing R'
      inst✝⁴ : CommRing S'
      inst✝³ : Algebra R R'
      inst✝² : Algebra S S'
      inst✝¹ : IsLocalization M R'
      inst✝ : IsLocalization (Submonoid.map f M) S'
      s : Set S
      hsone : Eq (Ideal.span s) Top.top
      hs : ∀ (t : S), Membership.mem s t → (fun {R S} [CommRing R] [CommRing S] => P …
      Mₐ : (a : ↑s) → Submonoid (Localization.Away ↑a) := fun a => Submonoid.map (al …
      Sₐ : ↑s → Type u := fun a => Localization (Mₐ a)
      hM : ∀ (a : ↑s), Eq (Submonoid.map ((algebraMap S (Localization.Away ↑a)).comp …
      this✝² : ∀ (a : ↑s), IsLocalization (Submonoid.map ((algebraMap S (Localizatio …
      this✝¹ : ∀ (a : ↑s), IsLocalization (Algebra.algebraMapSubmonoid (Localization …
      this✝ : (a : ↑s) → Algebra S' (Sₐ a) := fun a => (IsLocalization.map (Sₐ a) (a …
      this : ∀ (a : ↑s), IsScalarTower S S' (Sₐ a)
      ⊢ Eq (Ideal.span (Set.range fun a => (algebraMap S S') ↑a)) Top.top
    -/
  · rw [← Set.image_eq_range, ← Ideal.map_span, hsone, Ideal.map_top]
    /-
      🎉 no goals
    -/
  · convert IsLocalization.commutes (T := Sₐ a) (M₁ := M.map f) (S₁ := S')
      (S₂ := Localization.Away a.val) (M₂ := Submonoid.powers a.val)
    /-
      case h.e.h.e'_3.h
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hPl : RingHom.LocalizationPreserves fun {R S} [CommRing R] [CommRing S] => P
      R S : Type u
      inst✝⁷ : CommRing R
      inst✝⁶ : CommRing S
      f : RingHom R S
      M : Submonoid R
      R' S' : Type u
      inst✝⁵ : CommRing R'
      inst✝⁴ : CommRing S'
      inst✝³ : Algebra R R'
      inst✝² : Algebra S S'
      inst✝¹ : IsLocalization M R'
      inst✝ : IsLocalization (Submonoid.map f M) S'
      s : Set S
      hsone : Eq (Ideal.span s) Top.top
      hs : ∀ (t : S), Membership.mem s t → (fun {R S} [CommRing R] [CommRing S] => P …
      Mₐ : (a : ↑s) → Submonoid (Localization.Away ↑a) := fun a => Submonoid.map (al …
      Sₐ : ↑s → Type u := fun a => Localization (Mₐ a)
      hM : ∀ (a : ↑s), Eq (Submonoid.map ((algebraMap S (Localization.Away ↑a)).comp …
      this✝² : ∀ (a : ↑s), IsLocalization (Submonoid.map ((algebraMap S (Localizatio …
      this✝¹ : ∀ (a : ↑s), IsLocalization (Algebra.algebraMapSubmonoid (Localization …
      this✝ : (a : ↑s) → Algebra S' (Sₐ a) := fun a => (IsLocalization.map (Sₐ a) (a …
      this : ∀ (a : ↑s), IsScalarTower S S' (Sₐ a)
      a : ↑s
      ⊢ Eq (Submonoid.powers ((fun a => (algebraMap S S') ↑a) a)) (Algebra.algebraMa …
    -/
    simp [Algebra.algebraMapSubmonoid]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_3
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hPl : RingHom.LocalizationPreserves fun {R S} [CommRing R] [CommRing S] => P
      R S : Type u
      inst✝⁷ : CommRing R
      inst✝⁶ : CommRing S
      f : RingHom R S
      M : Submonoid R
      R' S' : Type u
      inst✝⁵ : CommRing R'
      inst✝⁴ : CommRing S'
      inst✝³ : Algebra R R'
      inst✝² : Algebra S S'
      inst✝¹ : IsLocalization M R'
      inst✝ : IsLocalization (Submonoid.map f M) S'
      s : Set S
      hsone : Eq (Ideal.span s) Top.top
      hs : ∀ (t : S), Membership.mem s t → (fun {R S} [CommRing R] [CommRing S] => P …
      Mₐ : (a : ↑s) → Submonoid (Localization.Away ↑a) := fun a => Submonoid.map (al …
      Sₐ : ↑s → Type u := fun a => Localization (Mₐ a)
      hM : ∀ (a : ↑s), Eq (Submonoid.map ((algebraMap S (Localization.Away ↑a)).comp …
      this✝² : ∀ (a : ↑s), IsLocalization (Submonoid.map ((algebraMap S (Localizatio …
      this✝¹ : ∀ (a : ↑s), IsLocalization (Algebra.algebraMapSubmonoid (Localization …
      this✝ : (a : ↑s) → Algebra S' (Sₐ a) := fun a => (IsLocalization.map (Sₐ a) (a …
      this : ∀ (a : ↑s), IsScalarTower S S' (Sₐ a)
      a : ↑s
      ⊢ P ((algebraMap S' (Sₐ a)).comp (IsLocalization.map S' f ⋯))
    -/
  · rw [algebraMap_toAlgebra, IsLocalization.map_comp_map]
    /-
      case intro.intro.refine_3
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hPl : RingHom.LocalizationPreserves fun {R S} [CommRing R] [CommRing S] => P
      R S : Type u
      inst✝⁷ : CommRing R
      inst✝⁶ : CommRing S
      f : RingHom R S
      M : Submonoid R
      R' S' : Type u
      inst✝⁵ : CommRing R'
      inst✝⁴ : CommRing S'
      inst✝³ : Algebra R R'
      inst✝² : Algebra S S'
      inst✝¹ : IsLocalization M R'
      inst✝ : IsLocalization (Submonoid.map f M) S'
      s : Set S
      hsone : Eq (Ideal.span s) Top.top
      hs : ∀ (t : S), Membership.mem s t → (fun {R S} [CommRing R] [CommRing S] => P …
      Mₐ : (a : ↑s) → Submonoid (Localization.Away ↑a) := fun a => Submonoid.map (al …
      Sₐ : ↑s → Type u := fun a => Localization (Mₐ a)
      hM : ∀ (a : ↑s), Eq (Submonoid.map ((algebraMap S (Localization.Away ↑a)).comp …
      this✝² : ∀ (a : ↑s), IsLocalization (Submonoid.map ((algebraMap S (Localizatio …
      this✝¹ : ∀ (a : ↑s), IsLocalization (Algebra.algebraMapSubmonoid (Localization …
      this✝ : (a : ↑s) → Algebra S' (Sₐ a) := fun a => (IsLocalization.map (Sₐ a) (a …
      this : ∀ (a : ↑s), IsScalarTower S S' (Sₐ a)
      a : ↑s
      ⊢ P (IsLocalization.map (Sₐ a) ((algebraMap S (Localization.Away ↑a)).comp f) ⋯)
    -/
    apply hPl
    /-
      case intro.intro.refine_3.a
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hPl : RingHom.LocalizationPreserves fun {R S} [CommRing R] [CommRing S] => P
      R S : Type u
      inst✝⁷ : CommRing R
      inst✝⁶ : CommRing S
      f : RingHom R S
      M : Submonoid R
      R' S' : Type u
      inst✝⁵ : CommRing R'
      inst✝⁴ : CommRing S'
      inst✝³ : Algebra R R'
      inst✝² : Algebra S S'
      inst✝¹ : IsLocalization M R'
      inst✝ : IsLocalization (Submonoid.map f M) S'
      s : Set S
      hsone : Eq (Ideal.span s) Top.top
      hs : ∀ (t : S), Membership.mem s t → (fun {R S} [CommRing R] [CommRing S] => P …
      Mₐ : (a : ↑s) → Submonoid (Localization.Away ↑a) := fun a => Submonoid.map (al …
      Sₐ : ↑s → Type u := fun a => Localization (Mₐ a)
      hM : ∀ (a : ↑s), Eq (Submonoid.map ((algebraMap S (Localization.Away ↑a)).comp …
      this✝² : ∀ (a : ↑s), IsLocalization (Submonoid.map ((algebraMap S (Localizatio …
      this✝¹ : ∀ (a : ↑s), IsLocalization (Algebra.algebraMapSubmonoid (Localization …
      this✝ : (a : ↑s) → Algebra S' (Sₐ a) := fun a => (IsLocalization.map (Sₐ a) (a …
      this : ∀ (a : ↑s), IsScalarTower S S' (Sₐ a)
      a : ↑s
      ⊢ P ((algebraMap S (Localization.Away ↑a)).comp f)
    -/
    exact hs a.val a.property
    /-
      🎉 no goals
    -/


/-- If `P` is preserved by localizations and stable under composition with localization
away maps, then `Locally P` is a local property of ring homomorphisms. -/
lemma locally_propertyIsLocal (hPl : LocalizationAwayPreserves P)
    (hPa : StableUnderCompositionWithLocalizationAway P) : PropertyIsLocal (Locally P) where
  localizationAwayPreserves := locally_localizationAwayPreserves hPl
  StableUnderCompositionWithLocalizationAwayTarget :=
    locally_StableUnderCompositionWithLocalizationAwayTarget hPl.respectsIso hPa.right
  ofLocalizationSpan := (locally_ofLocalizationSpanTarget hPl.respectsIso).ofLocalizationSpan
    (locally_StableUnderCompositionWithLocalizationAwaySource hPa.left)
  ofLocalizationSpanTarget := locally_ofLocalizationSpanTarget hPl.respectsIso


