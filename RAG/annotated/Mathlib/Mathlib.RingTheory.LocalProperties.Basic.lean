/-- A property `P` of comm rings is said to be preserved by localization
  if `P` holds for `M⁻¹R` whenever `P` holds for `R`. -/
def LocalizationPreserves : Prop :=
  ∀ {R : Type u} [hR : CommRing R] (M : Submonoid R) (S : Type u) [hS : CommRing S] [Algebra R S]
    [IsLocalization M S], @P R hR → @P S hS


/-- A property `P` of comm rings satisfies `OfLocalizationMaximal`
  if `P` holds for `R` whenever `P` holds for `Rₘ` for all maximal ideal `m`. -/
def OfLocalizationMaximal : Prop :=
  ∀ (R : Type u) [CommRing R],
    (∀ (J : Ideal R) (_ : J.IsMaximal), P (Localization.AtPrime J)) → P R


/-- A property `P` of ring homs is said to contain identities if `P` holds
for the identity homomorphism of every ring. -/
def RingHom.ContainsIdentities := ∀ (R : Type u) [CommRing R], P (RingHom.id R)


/-- A property `P` of ring homs is said to be preserved by localization
 if `P` holds for `M⁻¹R →+* M⁻¹S` whenever `P` holds for `R →+* S`. -/
def RingHom.LocalizationPreserves :=
  ∀ ⦃R S : Type u⦄ [CommRing R] [CommRing S] (f : R →+* S) (M : Submonoid R) (R' S' : Type u)
    [CommRing R'] [CommRing S'] [Algebra R R'] [Algebra S S'] [IsLocalization M R']
    [IsLocalization (M.map f) S'],
    P f → P (IsLocalization.map S' f (Submonoid.le_comap_map M) : R' →+* S')


/-- A property `P` of ring homs is said to be preserved by localization away
 if `P` holds for `Rᵣ →+* Sᵣ` whenever `P` holds for `R →+* S`. -/
def RingHom.LocalizationAwayPreserves :=
  ∀ ⦃R S : Type u⦄ [CommRing R] [CommRing S] (f : R →+* S) (r : R) (R' S' : Type u)
    [CommRing R'] [CommRing S'] [Algebra R R'] [Algebra S S'] [IsLocalization.Away r R']
    [IsLocalization.Away (f r) S'],
    P f → P (IsLocalization.Away.map R' S' f r : R' →+* S')


/-- A property `P` of ring homs satisfies `RingHom.OfLocalizationFiniteSpan`
if `P` holds for `R →+* S` whenever there exists a finite set `{ r }` that spans `R` such that
`P` holds for `Rᵣ →+* Sᵣ`.

Note that this is equivalent to `RingHom.OfLocalizationSpan` via
`RingHom.ofLocalizationSpan_iff_finite`, but this is easier to prove. -/
def RingHom.OfLocalizationFiniteSpan :=
  ∀ ⦃R S : Type u⦄ [CommRing R] [CommRing S] (f : R →+* S) (s : Finset R)
    (_ : Ideal.span (s : Set R) = ⊤) (_ : ∀ r : s, P (Localization.awayMap f r)), P f


/-- A property `P` of ring homs satisfies `RingHom.OfLocalizationFiniteSpan`
if `P` holds for `R →+* S` whenever there exists a set `{ r }` that spans `R` such that
`P` holds for `Rᵣ →+* Sᵣ`.

Note that this is equivalent to `RingHom.OfLocalizationFiniteSpan` via
`RingHom.ofLocalizationSpan_iff_finite`, but this has less restrictions when applying. -/
def RingHom.OfLocalizationSpan :=
  ∀ ⦃R S : Type u⦄ [CommRing R] [CommRing S] (f : R →+* S) (s : Set R) (_ : Ideal.span s = ⊤)
    (_ : ∀ r : s, P (Localization.awayMap f r)), P f


/-- A property `P` of ring homs satisfies `RingHom.HoldsForLocalizationAway`
 if `P` holds for each localization map `R →+* Rᵣ`. -/
def RingHom.HoldsForLocalizationAway : Prop :=
  ∀ ⦃R : Type u⦄ (S : Type u) [CommRing R] [CommRing S] [Algebra R S] (r : R)
    [IsLocalization.Away r S], P (algebraMap R S)


/-- A property `P` of ring homs satisfies `RingHom.StableUnderCompositionWithLocalizationAwaySource`
if whenever `P` holds for `f` it also holds for the composition with
localization maps on the source. -/
def RingHom.StableUnderCompositionWithLocalizationAwaySource : Prop :=
  ∀ ⦃R : Type u⦄ (S : Type u) ⦃T : Type u⦄ [CommRing R] [CommRing S] [CommRing T] [Algebra R S]
    (r : R) [IsLocalization.Away r S] (f : S →+* T), P f → P (f.comp (algebraMap R S))


/-- A property `P` of ring homs satisfies `RingHom.StableUnderCompositionWithLocalizationAway`
if whenever `P` holds for `f` it also holds for the composition with
localization maps on the target. -/
def RingHom.StableUnderCompositionWithLocalizationAwayTarget : Prop :=
  ∀ ⦃R S : Type u⦄ (T : Type u) [CommRing R] [CommRing S] [CommRing T] [Algebra S T] (s : S)
    [IsLocalization.Away s T] (f : R →+* S), P f → P ((algebraMap S T).comp f)


/-- A property `P` of ring homs satisfies `RingHom.StableUnderCompositionWithLocalizationAway`
if whenever `P` holds for `f` it also holds for the composition with
localization maps on the left and on the right. -/
def RingHom.StableUnderCompositionWithLocalizationAway : Prop :=
  StableUnderCompositionWithLocalizationAwaySource P ∧
    StableUnderCompositionWithLocalizationAwayTarget P


/-- A property `P` of ring homs satisfies `RingHom.OfLocalizationFiniteSpanTarget`
if `P` holds for `R →+* S` whenever there exists a finite set `{ r }` that spans `S` such that
`P` holds for `R →+* Sᵣ`.

Note that this is equivalent to `RingHom.OfLocalizationSpanTarget` via
`RingHom.ofLocalizationSpanTarget_iff_finite`, but this is easier to prove. -/
def RingHom.OfLocalizationFiniteSpanTarget : Prop :=
  ∀ ⦃R S : Type u⦄ [CommRing R] [CommRing S] (f : R →+* S) (s : Finset S)
    (_ : Ideal.span (s : Set S) = ⊤)
    (_ : ∀ r : s, P ((algebraMap S (Localization.Away (r : S))).comp f)), P f


/-- A property `P` of ring homs satisfies `RingHom.OfLocalizationSpanTarget`
if `P` holds for `R →+* S` whenever there exists a set `{ r }` that spans `S` such that
`P` holds for `R →+* Sᵣ`.

Note that this is equivalent to `RingHom.OfLocalizationFiniteSpanTarget` via
`RingHom.ofLocalizationSpanTarget_iff_finite`, but this has less restrictions when applying. -/
def RingHom.OfLocalizationSpanTarget : Prop :=
  ∀ ⦃R S : Type u⦄ [CommRing R] [CommRing S] (f : R →+* S) (s : Set S) (_ : Ideal.span s = ⊤)
    (_ : ∀ r : s, P ((algebraMap S (Localization.Away (r : S))).comp f)), P f


/-- A property `P` of ring homs satisfies `RingHom.OfLocalizationPrime`
if `P` holds for `R` whenever `P` holds for `Rₘ` for all prime ideals `p`. -/
def RingHom.OfLocalizationPrime : Prop :=
  ∀ ⦃R S : Type u⦄ [CommRing R] [CommRing S] (f : R →+* S),
    (∀ (J : Ideal S) (_ : J.IsPrime), P (Localization.localRingHom _ J f rfl)) → P f


/-- A property of ring homs is local if it is preserved by localizations and compositions, and for
each `{ r }` that spans `S`, we have `P (R →+* S) ↔ ∀ r, P (R →+* Sᵣ)`. -/
structure RingHom.PropertyIsLocal : Prop where
  localizationAwayPreserves : RingHom.LocalizationAwayPreserves @P
  ofLocalizationSpanTarget : RingHom.OfLocalizationSpanTarget @P
  ofLocalizationSpan : RingHom.OfLocalizationSpan @P
  StableUnderCompositionWithLocalizationAwayTarget :
    RingHom.StableUnderCompositionWithLocalizationAwayTarget @P


theorem RingHom.ofLocalizationSpan_iff_finite :
    RingHom.OfLocalizationSpan @P ↔ RingHom.OfLocalizationFiniteSpan @P := by
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    ⊢ Iff (RingHom.OfLocalizationSpan P) (RingHom.OfLocalizationFiniteSpan P)
  -/
  delta RingHom.OfLocalizationSpan RingHom.OfLocalizationFiniteSpan
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    ⊢ Iff (∀ ⦃R S : Type u⦄ [inst : CommRing R] [inst_1 : CommRing S] (f : RingHom …
  -/
  apply forall₅_congr
  -- TODO: Using `refine` here breaks `resetI`.
  /-
    case h
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    ⊢ ∀ (a b : Type u) (c : CommRing a) (d : CommRing b) (e : RingHom a b), Iff (∀ …
  -/
  intros
  /-
    case h
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    a✝ b✝ : Type u
    c✝ : CommRing a✝
    d✝ : CommRing b✝
    e✝ : RingHom a✝ b✝
    ⊢ Iff (∀ (s : Set a✝), Eq (Ideal.span s) Top.top → (∀ (r : ↑s), P (Localizatio …
  -/
  constructor
    /-
      case h.mp
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      a✝ b✝ : Type u
      c✝ : CommRing a✝
      d✝ : CommRing b✝
      e✝ : RingHom a✝ b✝
      ⊢ (∀ (s : Set a✝), Eq (Ideal.span s) Top.top → (∀ (r : ↑s), P (Localization.aw …
    -/
  · intro h s; exact h s
               /-
                 🎉 no goals
               -/
    /-
      case h.mpr
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      a✝ b✝ : Type u
      c✝ : CommRing a✝
      d✝ : CommRing b✝
      e✝ : RingHom a✝ b✝
      ⊢ (∀ (s : Finset a✝), Eq (Ideal.span ↑s) Top.top → (∀ (r : Subtype fun x => Me …
    -/
  · intro h s hs hs'
    /-
      case h.mpr
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      a✝ b✝ : Type u
      c✝ : CommRing a✝
      d✝ : CommRing b✝
      e✝ : RingHom a✝ b✝
      h : ∀ (s : Finset a✝), Eq (Ideal.span ↑s) Top.top → (∀ (r : Subtype fun x => M …
      s : Set a✝
      hs : Eq (Ideal.span s) Top.top
      hs' : ∀ (r : ↑s), P (Localization.awayMap e✝ ↑r)
      ⊢ P e✝
    -/
    obtain ⟨s', h₁, h₂⟩ := (Ideal.span_eq_top_iff_finite s).mp hs
    /-
      case h.mpr.intro.intro
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      a✝ b✝ : Type u
      c✝ : CommRing a✝
      d✝ : CommRing b✝
      e✝ : RingHom a✝ b✝
      h : ∀ (s : Finset a✝), Eq (Ideal.span ↑s) Top.top → (∀ (r : Subtype fun x => M …
      s : Set a✝
      hs : Eq (Ideal.span s) Top.top
      hs' : ∀ (r : ↑s), P (Localization.awayMap e✝ ↑r)
      s' : Finset a✝
      h₁ : HasSubset.Subset (↑s') s
      h₂ : Eq (Ideal.span ↑s') Top.top
      ⊢ P e✝
    -/
    exact h s' h₂ fun x => hs' ⟨_, h₁ x.prop⟩
    /-
      🎉 no goals
    -/


theorem RingHom.ofLocalizationSpanTarget_iff_finite :
    RingHom.OfLocalizationSpanTarget @P ↔ RingHom.OfLocalizationFiniteSpanTarget @P := by
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    ⊢ Iff (RingHom.OfLocalizationSpanTarget P) (RingHom.OfLocalizationFiniteSpanTa …
  -/
  delta RingHom.OfLocalizationSpanTarget RingHom.OfLocalizationFiniteSpanTarget
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    ⊢ Iff (∀ ⦃R S : Type u⦄ [inst : CommRing R] [inst_1 : CommRing S] (f : RingHom …
  -/
  apply forall₅_congr
  -- TODO: Using `refine` here breaks `resetI`.
  /-
    case h
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    ⊢ ∀ (a b : Type u) (c : CommRing a) (d : CommRing b) (e : RingHom a b), Iff (∀ …
  -/
  intros
  /-
    case h
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    a✝ b✝ : Type u
    c✝ : CommRing a✝
    d✝ : CommRing b✝
    e✝ : RingHom a✝ b✝
    ⊢ Iff (∀ (s : Set b✝), Eq (Ideal.span s) Top.top → (∀ (r : ↑s), P ((algebraMap …
  -/
  constructor
    /-
      case h.mp
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      a✝ b✝ : Type u
      c✝ : CommRing a✝
      d✝ : CommRing b✝
      e✝ : RingHom a✝ b✝
      ⊢ (∀ (s : Set b✝), Eq (Ideal.span s) Top.top → (∀ (r : ↑s), P ((algebraMap b✝  …
    -/
  · intro h s; exact h s
               /-
                 🎉 no goals
               -/
    /-
      case h.mpr
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      a✝ b✝ : Type u
      c✝ : CommRing a✝
      d✝ : CommRing b✝
      e✝ : RingHom a✝ b✝
      ⊢ (∀ (s : Finset b✝), Eq (Ideal.span ↑s) Top.top → (∀ (r : Subtype fun x => Me …
    -/
  · intro h s hs hs'
    /-
      case h.mpr
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      a✝ b✝ : Type u
      c✝ : CommRing a✝
      d✝ : CommRing b✝
      e✝ : RingHom a✝ b✝
      h : ∀ (s : Finset b✝), Eq (Ideal.span ↑s) Top.top → (∀ (r : Subtype fun x => M …
      s : Set b✝
      hs : Eq (Ideal.span s) Top.top
      hs' : ∀ (r : ↑s), P ((algebraMap b✝ (Localization.Away ↑r)).comp e✝)
      ⊢ P e✝
    -/
    obtain ⟨s', h₁, h₂⟩ := (Ideal.span_eq_top_iff_finite s).mp hs
    /-
      case h.mpr.intro.intro
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      a✝ b✝ : Type u
      c✝ : CommRing a✝
      d✝ : CommRing b✝
      e✝ : RingHom a✝ b✝
      h : ∀ (s : Finset b✝), Eq (Ideal.span ↑s) Top.top → (∀ (r : Subtype fun x => M …
      s : Set b✝
      hs : Eq (Ideal.span s) Top.top
      hs' : ∀ (r : ↑s), P ((algebraMap b✝ (Localization.Away ↑r)).comp e✝)
      s' : Finset b✝
      h₁ : HasSubset.Subset (↑s') s
      h₂ : Eq (Ideal.span ↑s') Top.top
      ⊢ P e✝
    -/
    exact h s' h₂ fun x => hs' ⟨_, h₁ x.prop⟩
    /-
      🎉 no goals
    -/


theorem RingHom.HoldsForLocalizationAway.of_bijective
    (H : RingHom.HoldsForLocalizationAway P) (hf : Function.Bijective f) :
    P f := by
  /-
    R S : Type u
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    H : RingHom.HoldsForLocalizationAway fun {R S} [CommRing R] [CommRing S] => P
    hf : Function.Bijective ⇑f
    ⊢ P f
  -/
  letI := f.toAlgebra
  /-
    R S : Type u
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    H : RingHom.HoldsForLocalizationAway fun {R S} [CommRing R] [CommRing S] => P
    hf : Function.Bijective ⇑f
    this : Algebra R S := f.toAlgebra
    ⊢ P f
  -/
  have := IsLocalization.at_units (.powers (1 : R)) (by simp)
  have := IsLocalization.isLocalization_of_algEquiv (.powers (1 : R))
    (AlgEquiv.ofBijective (Algebra.ofId R S) hf)
  /-
    R S : Type u
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    H : RingHom.HoldsForLocalizationAway fun {R S} [CommRing R] [CommRing S] => P
    hf : Function.Bijective ⇑f
    this✝¹ : Algebra R S := f.toAlgebra
    this✝ : IsLocalization (Submonoid.powers 1) R
    this : IsLocalization (Submonoid.powers 1) S
    ⊢ P f
  -/
  exact H _ 1
  /-
    🎉 no goals
  -/


lemma RingHom.StableUnderComposition.stableUnderCompositionWithLocalizationAway
    (hPc : RingHom.StableUnderComposition P) (hPl : HoldsForLocalizationAway P) :
    StableUnderCompositionWithLocalizationAway P := by
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hPc : RingHom.StableUnderComposition fun {R S} [CommRing R] [CommRing S] => P
    hPl : RingHom.HoldsForLocalizationAway fun {R S} [CommRing R] [CommRing S] => P
    ⊢ RingHom.StableUnderCompositionWithLocalizationAway fun {R S} [CommRing R] [C …
  -/
  constructor
    /-
      case left
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hPc : RingHom.StableUnderComposition fun {R S} [CommRing R] [CommRing S] => P
      hPl : RingHom.HoldsForLocalizationAway fun {R S} [CommRing R] [CommRing S] => P
      ⊢ RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommRing …
    -/
  · introv _ _ hf
    /-
      case left
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hPc : RingHom.StableUnderComposition fun {R S} [CommRing R] [CommRing S] => P
      hPl : RingHom.HoldsForLocalizationAway fun {R S} [CommRing R] [CommRing S] => P
      R✝ S T : Type u
      inst✝⁴ : CommRing R✝
      inst✝³ : CommRing S
      inst✝² : CommRing T
      inst✝¹ : Algebra R✝ S
      r : R✝
      inst✝ : IsLocalization.Away r S
      f : RingHom S T
      hf : (fun {R S} [CommRing R] [CommRing S] => P) f
      ⊢ (fun {R S} [CommRing R] [CommRing S] => P) (f.comp (algebraMap R✝ S))
    -/
    exact hPc _ _ (hPl S r) hf
    /-
      🎉 no goals
    -/
    /-
      case right
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hPc : RingHom.StableUnderComposition fun {R S} [CommRing R] [CommRing S] => P
      hPl : RingHom.HoldsForLocalizationAway fun {R S} [CommRing R] [CommRing S] => P
      ⊢ RingHom.StableUnderCompositionWithLocalizationAwayTarget fun {R S} [CommRing …
    -/
  · introv _ _ hf
    /-
      case right
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hPc : RingHom.StableUnderComposition fun {R S} [CommRing R] [CommRing S] => P
      hPl : RingHom.HoldsForLocalizationAway fun {R S} [CommRing R] [CommRing S] => P
      R✝ S T : Type u
      inst✝⁴ : CommRing R✝
      inst✝³ : CommRing S
      inst✝² : CommRing T
      inst✝¹ : Algebra S T
      s : S
      inst✝ : IsLocalization.Away s T
      f : RingHom R✝ S
      hf : (fun {R S} [CommRing R] [CommRing S] => P) f
      ⊢ (fun {R S} [CommRing R] [CommRing S] => P) ((algebraMap S T).comp f)
    -/
    exact hPc _ _ hf (hPl T s)
    /-
      🎉 no goals
    -/


lemma RingHom.HoldsForLocalizationAway.containsIdentities (hPl : HoldsForLocalizationAway P) :
    ContainsIdentities P := by
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hPl : RingHom.HoldsForLocalizationAway fun {R S} [CommRing R] [CommRing S] => P
    ⊢ RingHom.ContainsIdentities fun {R S} [CommRing R] [CommRing S] => P
  -/
  introv R
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hPl : RingHom.HoldsForLocalizationAway fun {R S} [CommRing R] [CommRing S] => P
    R : Type u
    inst✝ : CommRing R
    ⊢ P (RingHom.id R)
  -/
  exact hPl.of_bijective _ _ Function.bijective_id
  /-
    🎉 no goals
  -/


lemma RingHom.LocalizationAwayPreserves.respectsIso
    (hP : LocalizationAwayPreserves P) :
    RespectsIso P where
  left {R S T} _ _ _ f e hf := by
    /-
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => P
      R S T : Type u
      x✝² : CommRing R
      x✝¹ : CommRing S
      x✝ : CommRing T
      f : RingHom R S
      e : RingEquiv S T
      hf : (fun {R S} [CommRing R] [CommRing S] => P) f
      ⊢ (fun {R S} [CommRing R] [CommRing S] => P) (e.toRingHom.comp f)
    -/
    letI := e.toRingHom.toAlgebra
    have : IsLocalization.Away (1 : R) R :=
      IsLocalization.away_of_isUnit_of_bijective _ isUnit_one (Equiv.refl _).bijective
    have : IsLocalization.Away (f 1) T :=
      IsLocalization.away_of_isUnit_of_bijective _ (by simp) e.bijective
    /-
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => P
      R S T : Type u
      x✝² : CommRing R
      x✝¹ : CommRing S
      x✝ : CommRing T
      f : RingHom R S
      e : RingEquiv S T
      hf : (fun {R S} [CommRing R] [CommRing S] => P) f
      this✝¹ : Algebra S T := e.toRingHom.toAlgebra
      this✝ : IsLocalization.Away 1 R
      this : IsLocalization.Away (f 1) T
      ⊢ (fun {R S} [CommRing R] [CommRing S] => P) (e.toRingHom.comp f)
    -/
    convert hP f 1 R T hf
    /-
      case h.e'_5
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => P
      R S T : Type u
      x✝² : CommRing R
      x✝¹ : CommRing S
      x✝ : CommRing T
      f : RingHom R S
      e : RingEquiv S T
      hf : (fun {R S} [CommRing R] [CommRing S] => P) f
      this✝¹ : Algebra S T := e.toRingHom.toAlgebra
      this✝ : IsLocalization.Away 1 R
      this : IsLocalization.Away (f 1) T
      ⊢ Eq (e.toRingHom.comp f) (IsLocalization.Away.map R T f 1)
    -/
    trans (IsLocalization.Away.map R T f 1).comp (algebraMap R R)
      /-
        P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        hP : RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => P
        R S T : Type u
        x✝² : CommRing R
        x✝¹ : CommRing S
        x✝ : CommRing T
        f : RingHom R S
        e : RingEquiv S T
        hf : (fun {R S} [CommRing R] [CommRing S] => P) f
        this✝¹ : Algebra S T := e.toRingHom.toAlgebra
        this✝ : IsLocalization.Away 1 R
        this : IsLocalization.Away (f 1) T
        ⊢ Eq (e.toRingHom.comp f) ((IsLocalization.Away.map R T f 1).comp (algebraMap  …
      -/
    · rw [IsLocalization.Away.map, IsLocalization.map_comp]; rfl
                                                             /-
                                                               🎉 no goals
                                                             -/
      /-
        P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        hP : RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => P
        R S T : Type u
        x✝² : CommRing R
        x✝¹ : CommRing S
        x✝ : CommRing T
        f : RingHom R S
        e : RingEquiv S T
        hf : (fun {R S} [CommRing R] [CommRing S] => P) f
        this✝¹ : Algebra S T := e.toRingHom.toAlgebra
        this✝ : IsLocalization.Away 1 R
        this : IsLocalization.Away (f 1) T
        ⊢ Eq ((IsLocalization.Away.map R T f 1).comp (algebraMap R R)) (IsLocalization …
      -/
    · rfl
      /-
        🎉 no goals
      -/
  right {R S T} _ _ _ f e hf := by
    /-
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => P
      R S T : Type u
      x✝² : CommRing R
      x✝¹ : CommRing S
      x✝ : CommRing T
      f : RingHom S T
      e : RingEquiv R S
      hf : (fun {R S} [CommRing R] [CommRing S] => P) f
      ⊢ (fun {R S} [CommRing R] [CommRing S] => P) (f.comp e.toRingHom)
    -/
    letI := e.symm.toRingHom.toAlgebra
    have : IsLocalization.Away (1 : S) R :=
      IsLocalization.away_of_isUnit_of_bijective _ isUnit_one e.symm.bijective
    have : IsLocalization.Away (f 1) T :=
      IsLocalization.away_of_isUnit_of_bijective _ (by simp) (Equiv.refl _).bijective
    /-
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => P
      R S T : Type u
      x✝² : CommRing R
      x✝¹ : CommRing S
      x✝ : CommRing T
      f : RingHom S T
      e : RingEquiv R S
      hf : (fun {R S} [CommRing R] [CommRing S] => P) f
      this✝¹ : Algebra S R := e.symm.toRingHom.toAlgebra
      this✝ : IsLocalization.Away 1 R
      this : IsLocalization.Away (f 1) T
      ⊢ (fun {R S} [CommRing R] [CommRing S] => P) (f.comp e.toRingHom)
    -/
    convert hP f 1 R T hf
    have : (IsLocalization.Away.map R T f 1).comp e.symm.toRingHom = f :=
      IsLocalization.map_comp ..
    /-
      case h.e'_5
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => P
      R S T : Type u
      x✝² : CommRing R
      x✝¹ : CommRing S
      x✝ : CommRing T
      f : RingHom S T
      e : RingEquiv R S
      hf : (fun {R S} [CommRing R] [CommRing S] => P) f
      this✝² : Algebra S R := e.symm.toRingHom.toAlgebra
      this✝¹ : IsLocalization.Away 1 R
      this✝ : IsLocalization.Away (f 1) T
      this : Eq ((IsLocalization.Away.map R T f 1).comp e.symm.toRingHom) f
      ⊢ Eq (f.comp e.toRingHom) (IsLocalization.Away.map R T f 1)
    -/
    conv_lhs => rw [← this, RingHom.comp_assoc]
    /-
      case h.e'_5
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => P
      R S T : Type u
      x✝² : CommRing R
      x✝¹ : CommRing S
      x✝ : CommRing T
      f : RingHom S T
      e : RingEquiv R S
      hf : (fun {R S} [CommRing R] [CommRing S] => P) f
      this✝² : Algebra S R := e.symm.toRingHom.toAlgebra
      this✝¹ : IsLocalization.Away 1 R
      this✝ : IsLocalization.Away (f 1) T
      this : Eq ((IsLocalization.Away.map R T f 1).comp e.symm.toRingHom) f
      ⊢ Eq ((IsLocalization.Away.map R T f 1).comp (e.symm.toRingHom.comp e.toRingHo …
    -/
    simp only [RingEquiv.toRingHom_eq_coe, RingEquiv.symm_comp, RingHomCompTriple.comp_eq]
    /-
      🎉 no goals
    -/


lemma RingHom.StableUnderCompositionWithLocalizationAway.respectsIso
    (hP : StableUnderCompositionWithLocalizationAway P) :
    RespectsIso P where
  left {R S T} _ _ _ f e hf := by
    /-
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.StableUnderCompositionWithLocalizationAway fun {R S} [CommRing R] …
      R S T : Type u
      x✝² : CommRing R
      x✝¹ : CommRing S
      x✝ : CommRing T
      f : RingHom R S
      e : RingEquiv S T
      hf : (fun {R S} [CommRing R] [CommRing S] => P) f
      ⊢ (fun {R S} [CommRing R] [CommRing S] => P) (e.toRingHom.comp f)
    -/
    letI := e.toRingHom.toAlgebra
    have : IsLocalization.Away (1 : S) T :=
      IsLocalization.away_of_isUnit_of_bijective _ isUnit_one e.bijective
    /-
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.StableUnderCompositionWithLocalizationAway fun {R S} [CommRing R] …
      R S T : Type u
      x✝² : CommRing R
      x✝¹ : CommRing S
      x✝ : CommRing T
      f : RingHom R S
      e : RingEquiv S T
      hf : (fun {R S} [CommRing R] [CommRing S] => P) f
      this✝ : Algebra S T := e.toRingHom.toAlgebra
      this : IsLocalization.Away 1 T
      ⊢ (fun {R S} [CommRing R] [CommRing S] => P) (e.toRingHom.comp f)
    -/
    exact hP.right T (1 : S) f hf
    /-
      🎉 no goals
    -/
  right {R S T} _ _ _ f e hf := by
    /-
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.StableUnderCompositionWithLocalizationAway fun {R S} [CommRing R] …
      R S T : Type u
      x✝² : CommRing R
      x✝¹ : CommRing S
      x✝ : CommRing T
      f : RingHom S T
      e : RingEquiv R S
      hf : (fun {R S} [CommRing R] [CommRing S] => P) f
      ⊢ (fun {R S} [CommRing R] [CommRing S] => P) (f.comp e.toRingHom)
    -/
    letI := e.toRingHom.toAlgebra
    have : IsLocalization.Away (1 : R) S :=
      IsLocalization.away_of_isUnit_of_bijective _ isUnit_one e.bijective
    /-
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.StableUnderCompositionWithLocalizationAway fun {R S} [CommRing R] …
      R S T : Type u
      x✝² : CommRing R
      x✝¹ : CommRing S
      x✝ : CommRing T
      f : RingHom S T
      e : RingEquiv R S
      hf : (fun {R S} [CommRing R] [CommRing S] => P) f
      this✝ : Algebra R S := e.toRingHom.toAlgebra
      this : IsLocalization.Away 1 S
      ⊢ (fun {R S} [CommRing R] [CommRing S] => P) (f.comp e.toRingHom)
    -/
    exact hP.left S (1 : R) f hf
    /-
      🎉 no goals
    -/


theorem RingHom.PropertyIsLocal.respectsIso (hP : RingHom.PropertyIsLocal @P) :
    RingHom.RespectsIso @P :=
  hP.localizationAwayPreserves.respectsIso

-- Almost all arguments are implicit since this is not intended to use mid-proof.

theorem RingHom.LocalizationPreserves.away (H : RingHom.LocalizationPreserves @P) :
    RingHom.LocalizationAwayPreserves P := by
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    H : RingHom.LocalizationPreserves P
    ⊢ RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => P
  -/
  intros R S _ _ f r R' S' _ _ _ _ _ _ hf
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    H : RingHom.LocalizationPreserves P
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
    hf : P f
    ⊢ P (IsLocalization.Away.map R' S' f r)
  -/
  have : IsLocalization ((Submonoid.powers r).map f) S' := by rw [Submonoid.map_powers]; assumption
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    H : RingHom.LocalizationPreserves P
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
    hf : P f
    this : IsLocalization (Submonoid.map f (Submonoid.powers r)) S'
    ⊢ P (IsLocalization.Away.map R' S' f r)
  -/
  exact H f (Submonoid.powers r) R' S' hf
  /-
    🎉 no goals
  -/


lemma RingHom.PropertyIsLocal.HoldsForLocalizationAway (hP : RingHom.PropertyIsLocal @P)
    (hPi : ContainsIdentities P) :
    RingHom.HoldsForLocalizationAway @P := by
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.PropertyIsLocal P
    hPi : RingHom.ContainsIdentities fun {R S} [CommRing R] [CommRing S] => P
    ⊢ RingHom.HoldsForLocalizationAway P
  -/
  introv R _
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.PropertyIsLocal P
    hPi : RingHom.ContainsIdentities fun {R S} [CommRing R] [CommRing S] => P
    R S : Type u
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    r : R
    inst✝ : IsLocalization.Away r S
    ⊢ P (algebraMap R S)
  -/
  have : algebraMap R S = (algebraMap R S).comp (RingHom.id R) := by simp
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.PropertyIsLocal P
    hPi : RingHom.ContainsIdentities fun {R S} [CommRing R] [CommRing S] => P
    R S : Type u
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    r : R
    inst✝ : IsLocalization.Away r S
    this : Eq (algebraMap R S) ((algebraMap R S).comp (RingHom.id R))
    ⊢ P (algebraMap R S)
  -/
  rw [this]
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.PropertyIsLocal P
    hPi : RingHom.ContainsIdentities fun {R S} [CommRing R] [CommRing S] => P
    R S : Type u
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    r : R
    inst✝ : IsLocalization.Away r S
    this : Eq (algebraMap R S) ((algebraMap R S).comp (RingHom.id R))
    ⊢ P ((algebraMap R S).comp (RingHom.id R))
  -/
  apply hP.StableUnderCompositionWithLocalizationAwayTarget S r
  /-
    case a
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.PropertyIsLocal P
    hPi : RingHom.ContainsIdentities fun {R S} [CommRing R] [CommRing S] => P
    R S : Type u
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    r : R
    inst✝ : IsLocalization.Away r S
    this : Eq (algebraMap R S) ((algebraMap R S).comp (RingHom.id R))
    ⊢ P (RingHom.id R)
  -/
  apply hPi
  /-
    🎉 no goals
  -/


theorem RingHom.OfLocalizationSpanTarget.ofLocalizationSpan
    (hP : RingHom.OfLocalizationSpanTarget @P)
    (hP' : RingHom.StableUnderCompositionWithLocalizationAwaySource @P) :
    RingHom.OfLocalizationSpan @P := by
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.OfLocalizationSpanTarget P
    hP' : RingHom.StableUnderCompositionWithLocalizationAwaySource P
    ⊢ RingHom.OfLocalizationSpan P
  -/
  introv R hs hs'
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.OfLocalizationSpanTarget P
    hP' : RingHom.StableUnderCompositionWithLocalizationAwaySource P
    R S : Type u
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set R
    hs : Eq (Ideal.span s) Top.top
    hs' : ∀ (r : ↑s), P (Localization.awayMap f ↑r)
    ⊢ P f
  -/
  apply_fun Ideal.map f at hs
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.OfLocalizationSpanTarget P
    hP' : RingHom.StableUnderCompositionWithLocalizationAwaySource P
    R S : Type u
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set R
    hs' : ∀ (r : ↑s), P (Localization.awayMap f ↑r)
    hs : Eq (Ideal.map f (Ideal.span s)) (Ideal.map f Top.top)
    ⊢ P f
  -/
  rw [Ideal.map_span, Ideal.map_top] at hs
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.OfLocalizationSpanTarget P
    hP' : RingHom.StableUnderCompositionWithLocalizationAwaySource P
    R S : Type u
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set R
    hs' : ∀ (r : ↑s), P (Localization.awayMap f ↑r)
    hs : Eq (Ideal.span (Set.image (⇑f) s)) Top.top
    ⊢ P f
  -/
  apply hP _ _ hs
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.OfLocalizationSpanTarget P
    hP' : RingHom.StableUnderCompositionWithLocalizationAwaySource P
    R S : Type u
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set R
    hs' : ∀ (r : ↑s), P (Localization.awayMap f ↑r)
    hs : Eq (Ideal.span (Set.image (⇑f) s)) Top.top
    ⊢ ∀ (r : ↑(Set.image (⇑f) s)), P ((algebraMap S (Localization.Away ↑r)).comp f)
  -/
  rintro ⟨_, r, hr, rfl⟩
  rw [← IsLocalization.map_comp (M := Submonoid.powers r) (S := Localization.Away r)
    (T := Submonoid.powers (f r))]
    /-
      case mk.intro.intro
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.OfLocalizationSpanTarget P
      hP' : RingHom.StableUnderCompositionWithLocalizationAwaySource P
      R S : Type u
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      s : Set R
      hs' : ∀ (r : ↑s), P (Localization.awayMap f ↑r)
      hs : Eq (Ideal.span (Set.image (⇑f) s)) Top.top
      r : R
      hr : Membership.mem s r
      ⊢ P ((IsLocalization.map (Localization.Away ↑⟨f r, ⋯⟩) f ?mk.intro.intro).comp …
    -/
  · apply hP' _ r
    /-
      case mk.intro.intro.a
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.OfLocalizationSpanTarget P
      hP' : RingHom.StableUnderCompositionWithLocalizationAwaySource P
      R S : Type u
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      s : Set R
      hs' : ∀ (r : ↑s), P (Localization.awayMap f ↑r)
      hs : Eq (Ideal.span (Set.image (⇑f) s)) Top.top
      r : R
      hr : Membership.mem s r
      ⊢ P (IsLocalization.map (Localization.Away ↑⟨f r, ⋯⟩) f ?mk.intro.intro)
    -/
    exact hs' ⟨r, hr⟩
    /-
      🎉 no goals
    -/


lemma RingHom.OfLocalizationSpan.ofIsLocalization
    (hP : RingHom.OfLocalizationSpan P) (hPi : RingHom.RespectsIso P)
    {R S : Type u} [CommRing R] [CommRing S] (f : R →+* S) (s : Set R) (hs : Ideal.span s = ⊤)
    (hT : ∀ r : s, ∃ (Rᵣ Sᵣ : Type u) (_ : CommRing Rᵣ) (_ : CommRing Sᵣ)
      (_ : Algebra R Rᵣ) (_ : Algebra S Sᵣ) (_ : IsLocalization.Away r.val Rᵣ)
      (_ : IsLocalization.Away (f r.val) Sᵣ) (fᵣ : Rᵣ →+* Sᵣ)
      (_ : fᵣ.comp (algebraMap R Rᵣ) = (algebraMap S Sᵣ).comp f),
        P fᵣ) : P f := by
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => P
    hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
    R S : Type u
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set R
    hs : Eq (Ideal.span s) Top.top
    hT : ∀ (r : ↑s), Exists fun Rᵣ => Exists fun Sᵣ => Exists fun x => Exists fun  …
    ⊢ P f
  -/
  apply hP _ s hs
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => P
    hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
    R S : Type u
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set R
    hs : Eq (Ideal.span s) Top.top
    hT : ∀ (r : ↑s), Exists fun Rᵣ => Exists fun Sᵣ => Exists fun x => Exists fun  …
    ⊢ ∀ (r : ↑s), (fun {R S} [CommRing R] [CommRing S] => P) (Localization.awayMap …
  -/
  intro r
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => P
    hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
    R S : Type u
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set R
    hs : Eq (Ideal.span s) Top.top
    hT : ∀ (r : ↑s), Exists fun Rᵣ => Exists fun Sᵣ => Exists fun x => Exists fun  …
    r : ↑s
    ⊢ P (Localization.awayMap f ↑r)
  -/
  obtain ⟨Rᵣ, Sᵣ, _, _, _, _, _, _, fᵣ, hfᵣ, hf⟩ := hT r
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => P
    hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
    R S : Type u
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set R
    hs : Eq (Ideal.span s) Top.top
    hT : ∀ (r : ↑s), Exists fun Rᵣ => Exists fun Sᵣ => Exists fun x => Exists fun  …
    r : ↑s
    Rᵣ Sᵣ : Type u
    w✝⁵ : CommRing Rᵣ
    w✝⁴ : CommRing Sᵣ
    w✝³ : Algebra R Rᵣ
    w✝² : Algebra S Sᵣ
    w✝¹ : IsLocalization.Away (↑r) Rᵣ
    w✝ : IsLocalization.Away (f ↑r) Sᵣ
    fᵣ : RingHom Rᵣ Sᵣ
    hfᵣ : Eq (fᵣ.comp (algebraMap R Rᵣ)) ((algebraMap S Sᵣ).comp f)
    hf : P fᵣ
    ⊢ P (Localization.awayMap f ↑r)
  -/
  let e₁ := (Localization.algEquiv (.powers r.val) Rᵣ).toRingEquiv
  let e₂ := (IsLocalization.algEquiv (.powers (f r.val))
    (Localization (.powers (f r.val))) Sᵣ).symm.toRingEquiv
  have : Localization.awayMap f r.val =
      (e₂.toRingHom.comp fᵣ).comp e₁.toRingHom := by
    apply IsLocalization.ringHom_ext (.powers r.val)
    ext x
    have : fᵣ ((algebraMap R Rᵣ) x) = algebraMap S Sᵣ (f x) := by
      rw [← RingHom.comp_apply, hfᵣ, RingHom.comp_apply]
    simp [-AlgEquiv.symm_toRingEquiv, e₂, e₁, Localization.awayMap, IsLocalization.Away.map, this]
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => P
    hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
    R S : Type u
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set R
    hs : Eq (Ideal.span s) Top.top
    hT : ∀ (r : ↑s), Exists fun Rᵣ => Exists fun Sᵣ => Exists fun x => Exists fun  …
    r : ↑s
    Rᵣ Sᵣ : Type u
    w✝⁵ : CommRing Rᵣ
    w✝⁴ : CommRing Sᵣ
    w✝³ : Algebra R Rᵣ
    w✝² : Algebra S Sᵣ
    w✝¹ : IsLocalization.Away (↑r) Rᵣ
    w✝ : IsLocalization.Away (f ↑r) Sᵣ
    fᵣ : RingHom Rᵣ Sᵣ
    hfᵣ : Eq (fᵣ.comp (algebraMap R Rᵣ)) ((algebraMap S Sᵣ).comp f)
    hf : P fᵣ
    e₁ : RingEquiv (Localization (Submonoid.powers ↑r)) Rᵣ := (Localization.algEqu …
    e₂ : RingEquiv Sᵣ (Localization (Submonoid.powers (f ↑r))) := (IsLocalization. …
    this : Eq (Localization.awayMap f ↑r) ((e₂.toRingHom.comp fᵣ).comp e₁.toRingHom)
    ⊢ P (Localization.awayMap f ↑r)
  -/
  rw [this]
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => P
    hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
    R S : Type u
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set R
    hs : Eq (Ideal.span s) Top.top
    hT : ∀ (r : ↑s), Exists fun Rᵣ => Exists fun Sᵣ => Exists fun x => Exists fun  …
    r : ↑s
    Rᵣ Sᵣ : Type u
    w✝⁵ : CommRing Rᵣ
    w✝⁴ : CommRing Sᵣ
    w✝³ : Algebra R Rᵣ
    w✝² : Algebra S Sᵣ
    w✝¹ : IsLocalization.Away (↑r) Rᵣ
    w✝ : IsLocalization.Away (f ↑r) Sᵣ
    fᵣ : RingHom Rᵣ Sᵣ
    hfᵣ : Eq (fᵣ.comp (algebraMap R Rᵣ)) ((algebraMap S Sᵣ).comp f)
    hf : P fᵣ
    e₁ : RingEquiv (Localization (Submonoid.powers ↑r)) Rᵣ := (Localization.algEqu …
    e₂ : RingEquiv Sᵣ (Localization (Submonoid.powers (f ↑r))) := (IsLocalization. …
    this : Eq (Localization.awayMap f ↑r) ((e₂.toRingHom.comp fᵣ).comp e₁.toRingHom)
    ⊢ P ((e₂.toRingHom.comp fᵣ).comp e₁.toRingHom)
  -/
  apply hPi.right
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.x
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => P
    hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
    R S : Type u
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set R
    hs : Eq (Ideal.span s) Top.top
    hT : ∀ (r : ↑s), Exists fun Rᵣ => Exists fun Sᵣ => Exists fun x => Exists fun  …
    r : ↑s
    Rᵣ Sᵣ : Type u
    w✝⁵ : CommRing Rᵣ
    w✝⁴ : CommRing Sᵣ
    w✝³ : Algebra R Rᵣ
    w✝² : Algebra S Sᵣ
    w✝¹ : IsLocalization.Away (↑r) Rᵣ
    w✝ : IsLocalization.Away (f ↑r) Sᵣ
    fᵣ : RingHom Rᵣ Sᵣ
    hfᵣ : Eq (fᵣ.comp (algebraMap R Rᵣ)) ((algebraMap S Sᵣ).comp f)
    hf : P fᵣ
    e₁ : RingEquiv (Localization (Submonoid.powers ↑r)) Rᵣ := (Localization.algEqu …
    e₂ : RingEquiv Sᵣ (Localization (Submonoid.powers (f ↑r))) := (IsLocalization. …
    this : Eq (Localization.awayMap f ↑r) ((e₂.toRingHom.comp fᵣ).comp e₁.toRingHom)
    ⊢ P (e₂.toRingHom.comp fᵣ)
  -/
  apply hPi.left
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.x.x
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => P
    hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
    R S : Type u
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set R
    hs : Eq (Ideal.span s) Top.top
    hT : ∀ (r : ↑s), Exists fun Rᵣ => Exists fun Sᵣ => Exists fun x => Exists fun  …
    r : ↑s
    Rᵣ Sᵣ : Type u
    w✝⁵ : CommRing Rᵣ
    w✝⁴ : CommRing Sᵣ
    w✝³ : Algebra R Rᵣ
    w✝² : Algebra S Sᵣ
    w✝¹ : IsLocalization.Away (↑r) Rᵣ
    w✝ : IsLocalization.Away (f ↑r) Sᵣ
    fᵣ : RingHom Rᵣ Sᵣ
    hfᵣ : Eq (fᵣ.comp (algebraMap R Rᵣ)) ((algebraMap S Sᵣ).comp f)
    hf : P fᵣ
    e₁ : RingEquiv (Localization (Submonoid.powers ↑r)) Rᵣ := (Localization.algEqu …
    e₂ : RingEquiv Sᵣ (Localization (Submonoid.powers (f ↑r))) := (IsLocalization. …
    this : Eq (Localization.awayMap f ↑r) ((e₂.toRingHom.comp fᵣ).comp e₁.toRingHom)
    ⊢ P fᵣ
  -/
  exact hf
  /-
    🎉 no goals
  -/


/-- Variant of `RingHom.OfLocalizationSpan.ofIsLocalization` where
`fᵣ = IsLocalization.Away.map`. -/
lemma RingHom.OfLocalizationSpan.ofIsLocalization'
    (hP : RingHom.OfLocalizationSpan P) (hPi : RingHom.RespectsIso P)
    {R S : Type u} [CommRing R] [CommRing S] (f : R →+* S) (s : Set R) (hs : Ideal.span s = ⊤)
    (hT : ∀ r : s, ∃ (Rᵣ Sᵣ : Type u) (_ : CommRing Rᵣ) (_ : CommRing Sᵣ)
      (_ : Algebra R Rᵣ) (_ : Algebra S Sᵣ) (_ : IsLocalization.Away r.val Rᵣ)
      (_ : IsLocalization.Away (f r.val) Sᵣ),
        P (IsLocalization.Away.map Rᵣ Sᵣ f r)) : P f := by
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => P
    hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
    R S : Type u
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set R
    hs : Eq (Ideal.span s) Top.top
    hT : ∀ (r : ↑s), Exists fun Rᵣ => Exists fun Sᵣ => Exists fun x => Exists fun  …
    ⊢ P f
  -/
  apply hP.ofIsLocalization hPi _ s hs
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => P
    hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
    R S : Type u
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set R
    hs : Eq (Ideal.span s) Top.top
    hT : ∀ (r : ↑s), Exists fun Rᵣ => Exists fun Sᵣ => Exists fun x => Exists fun  …
    ⊢ ∀ (r : ↑s), Exists fun Rᵣ => Exists fun Sᵣ => Exists fun x => Exists fun x_1 …
  -/
  intro r
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => P
    hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
    R S : Type u
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set R
    hs : Eq (Ideal.span s) Top.top
    hT : ∀ (r : ↑s), Exists fun Rᵣ => Exists fun Sᵣ => Exists fun x => Exists fun  …
    r : ↑s
    ⊢ Exists fun Rᵣ => Exists fun Sᵣ => Exists fun x => Exists fun x_1 => Exists f …
  -/
  obtain ⟨Rᵣ, Sᵣ, _, _, _, _, _, _, hf⟩ := hT r
  exact ⟨Rᵣ, Sᵣ, inferInstance, inferInstance, inferInstance, inferInstance,
    inferInstance, inferInstance, IsLocalization.Away.map Rᵣ Sᵣ f r, IsLocalization.map_comp _, hf⟩


lemma RingHom.OfLocalizationSpanTarget.ofIsLocalization
    (hP : RingHom.OfLocalizationSpanTarget P) (hP' : RingHom.RespectsIso P)
    {R S : Type u} [CommRing R] [CommRing S] (f : R →+* S) (s : Set S) (hs : Ideal.span s = ⊤)
    (hT : ∀ r : s, ∃ (T : Type u) (_ : CommRing T) (_ : Algebra S T)
      (_ : IsLocalization.Away (r : S) T), P ((algebraMap S T).comp f)) : P f := by
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.OfLocalizationSpanTarget fun {R S} [CommRing R] [CommRing S] => P
    hP' : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
    R S : Type u
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set S
    hs : Eq (Ideal.span s) Top.top
    hT : ∀ (r : ↑s), Exists fun T => Exists fun x => Exists fun x_1 => Exists fun  …
    ⊢ P f
  -/
  apply hP _ s hs
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.OfLocalizationSpanTarget fun {R S} [CommRing R] [CommRing S] => P
    hP' : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
    R S : Type u
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set S
    hs : Eq (Ideal.span s) Top.top
    hT : ∀ (r : ↑s), Exists fun T => Exists fun x => Exists fun x_1 => Exists fun  …
    ⊢ ∀ (r : ↑s), (fun {R S} [CommRing R] [CommRing S] => P) ((algebraMap S (Local …
  -/
  intros r
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.OfLocalizationSpanTarget fun {R S} [CommRing R] [CommRing S] => P
    hP' : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
    R S : Type u
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set S
    hs : Eq (Ideal.span s) Top.top
    hT : ∀ (r : ↑s), Exists fun T => Exists fun x => Exists fun x_1 => Exists fun  …
    r : ↑s
    ⊢ P ((algebraMap S (Localization.Away ↑r)).comp f)
  -/
  obtain ⟨T, _, _, _, hT⟩ := hT r
  convert hP'.1 _
    (Localization.algEquiv (R := S) (Submonoid.powers (r : S)) T).symm.toRingEquiv hT
  rw [← RingHom.comp_assoc, RingEquiv.toRingHom_eq_coe, AlgEquiv.toRingEquiv_eq_coe,
    AlgEquiv.toRingEquiv_toRingHom, Localization.coe_algEquiv_symm, IsLocalization.map_comp,
    RingHom.comp_id]


/-- Let `S` be an `R`-algebra and `Sᵣ` and `Rᵣ` be the respective localizations at a submonoid
`M` of `R`. If `P` is stable under base change and `P` holds for `algebraMap R S`, then
`P` holds for `algebraMap Rᵣ Sᵣ`. -/
lemma RingHom.IsStableUnderBaseChange.of_isLocalization [Algebra R S] [Algebra R Sᵣ] [Algebra Rᵣ Sᵣ]
    [IsScalarTower R S Sᵣ] [IsScalarTower R Rᵣ Sᵣ]
    (M : Submonoid R) [IsLocalization M Rᵣ] [IsLocalization (Algebra.algebraMapSubmonoid S M) Sᵣ]
    (h : P (algebraMap R S)) : P (algebraMap Rᵣ Sᵣ) :=
  letI : Algebra.IsPushout R S Rᵣ Sᵣ := Algebra.isPushout_of_isLocalization M Rᵣ S Sᵣ
  hP R S Rᵣ Sᵣ h


/-- If `P` is stable under base change and holds for `f`, then `P` holds for `f` localized
at any submonoid `M` of `R`. -/
lemma RingHom.IsStableUnderBaseChange.isLocalization_map (M : Submonoid R) [IsLocalization M Rᵣ]
    (f : R →+* S) [IsLocalization (M.map f) Sᵣ] (hf : P f) :
    P (IsLocalization.map Sᵣ f M.le_comap_map : Rᵣ →+* Sᵣ) := by
  algebraize [f, IsLocalization.map (S := Rᵣ) Sᵣ f M.le_comap_map,
    (IsLocalization.map (S := Rᵣ) Sᵣ f M.le_comap_map).comp (algebraMap R Rᵣ)]
  haveI : IsScalarTower R S Sᵣ := IsScalarTower.of_algebraMap_eq'
    (IsLocalization.map_comp M.le_comap_map)
  haveI : IsLocalization (Algebra.algebraMapSubmonoid S M) Sᵣ :=
    inferInstanceAs <| IsLocalization (M.map f) Sᵣ
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.IsStableUnderBaseChange P
    R S Rᵣ Sᵣ : Type u
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    inst✝⁵ : CommRing Rᵣ
    inst✝⁴ : CommRing Sᵣ
    inst✝³ : Algebra R Rᵣ
    inst✝² : Algebra S Sᵣ
    M : Submonoid R
    inst✝¹ : IsLocalization M Rᵣ
    f : RingHom R S
    inst✝ : IsLocalization (Submonoid.map f M) Sᵣ
    hf : P f
    algInst✝² : Algebra R S := f.toAlgebra
    algInst✝¹ : Algebra Rᵣ Sᵣ := (IsLocalization.map Sᵣ f ⋯).toAlgebra
    algInst✝ : Algebra R Sᵣ := ((IsLocalization.map Sᵣ f ⋯).comp (algebraMap R Rᵣ) …
    scalarTowerInst✝ : IsScalarTower R Rᵣ Sᵣ := IsScalarTower.of_algebraMap_eq' (E …
    this✝ : IsScalarTower R S Sᵣ
    this : IsLocalization (Algebra.algebraMapSubmonoid S M) Sᵣ
    ⊢ P (IsLocalization.map Sᵣ f ⋯)
  -/
  apply hP.of_isLocalization M hf
  /-
    🎉 no goals
  -/


lemma RingHom.IsStableUnderBaseChange.localizationPreserves : LocalizationPreserves P := by
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.IsStableUnderBaseChange P
    ⊢ RingHom.LocalizationPreserves fun {R S} [CommRing R] [CommRing S] => P
  -/
  introv R hf
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.IsStableUnderBaseChange P
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
    hf : P f
    ⊢ P (IsLocalization.map S' f ⋯)
  -/
  exact hP.isLocalization_map _ _ hf
  /-
    🎉 no goals
  -/


theorem Ideal.localized'_eq_map (I : Ideal R) :
    haveI := (isLocalizedModule_iff_isLocalization' p S).mpr inferInstance
    Submodule.localized' S p (Algebra.linearMap R S) I = I.map (algebraMap R S) :=
  SetLike.ext fun x ↦ by
    simp_rw [Submodule.mem_localized', IsLocalization.mem_map_algebraMap_iff p,
      IsLocalizedModule.mk'_eq_iff, mul_comm x, eq_comm (a := _ * x), ← Algebra.smul_def,
      Prod.exists, Subtype.exists, ← exists_prop]
    /-
      R : Type u_1
      S : Type u_2
      inst✝³ : CommSemiring R
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      p : Submonoid R
      inst✝ : IsLocalization p S
      I : Ideal R
      x : S
      ⊢ Iff (Exists fun m => Exists fun _h => Exists fun a => Exists fun b => Eq ((A …
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem Ideal.localized₀_eq_restrictScalars_map (I : Ideal R) :
    Submodule.localized₀ p (Algebra.linearMap R S) I = (I.map (algebraMap R S)).restrictScalars R :=
  congr(Submodule.restrictScalars R $(localized'_eq_map S p I))


theorem Algebra.idealMap_eq_ofEq_comp_toLocalized₀ (I : Ideal R) :
    haveI := (isLocalizedModule_iff_isLocalization' p S).mpr inferInstance
    Algebra.idealMap S I =
      (LinearEquiv.ofEq _ _ <| Ideal.localized₀_eq_restrictScalars_map S p I).toLinearMap ∘ₗ
      Submodule.toLocalized₀ p (Algebra.linearMap R S) I :=
  rfl


theorem Ideal.mem_of_localization_maximal {r : R} {J : Ideal R}
    (h : ∀ (P : Ideal R) (_ : P.IsMaximal),
      algebraMap R _ r ∈ Ideal.map (algebraMap R (Localization.AtPrime P)) J) :
    r ∈ J :=
  Submodule.mem_of_localization_maximal _ _ _ _ fun P hP ↦ by
    /-
      R : Type u_1
      inst✝ : CommSemiring R
      r : R
      J : Ideal R
      h : ∀ (P : Ideal R) (x : P.IsMaximal), Membership.mem (Ideal.map (algebraMap R …
      P : Ideal R
      hP : P.IsMaximal
      ⊢ Membership.mem (Submodule.localized₀ P.primeCompl (?m.182227 P) J) ((?m.1822 …
    -/
    apply (localized'_eq_map (Localization.AtPrime P) P.primeCompl J).symm ▸ h P hP
    /-
      🎉 no goals
    -/


/-- Let `I J : Ideal R`. If the localization of `I` at each maximal ideal `P` is included in
the localization of `J` at `P`, then `I ≤ J`. -/
theorem Ideal.le_of_localization_maximal {I J : Ideal R}
    (h : ∀ (P : Ideal R) (_ : P.IsMaximal),
      Ideal.map (algebraMap R (Localization.AtPrime P)) I ≤
        Ideal.map (algebraMap R (Localization.AtPrime P)) J) :
    I ≤ J :=
  fun _ hm ↦ mem_of_localization_maximal fun P hP ↦ h P hP (mem_map_of_mem _ hm)


/-- Let `I J : Ideal R`. If the localization of `I` at each maximal ideal `P` is equal to
the localization of `J` at `P`, then `I = J`. -/
theorem Ideal.eq_of_localization_maximal {I J : Ideal R}
    (h : ∀ (P : Ideal R) (_ : P.IsMaximal),
      Ideal.map (algebraMap R (Localization.AtPrime P)) I =
        Ideal.map (algebraMap R (Localization.AtPrime P)) J) :
    I = J :=
  le_antisymm (le_of_localization_maximal fun P hP ↦ (h P hP).le)
    (le_of_localization_maximal fun P hP ↦ (h P hP).ge)


/-- An ideal is trivial if its localization at every maximal ideal is trivial. -/
theorem ideal_eq_bot_of_localization' (I : Ideal R)
    (h : ∀ (J : Ideal R) (_ : J.IsMaximal),
      Ideal.map (algebraMap R (Localization.AtPrime J)) I = ⊥) :
    I = ⊥ :=
                                                  /-
                                                    R : Type u_1
                                                    inst✝ : CommSemiring R
                                                    I : Ideal R
                                                    h : ∀ (J : Ideal R) (x : J.IsMaximal), Eq (Ideal.map (algebraMap R (Localizati …
                                                    P : Ideal R
                                                    hP : P.IsMaximal
                                                    ⊢ Eq (Ideal.map (algebraMap R (Localization.AtPrime P)) I) (Ideal.map (algebra …
                                                  -/
  Ideal.eq_of_localization_maximal fun P hP => by simpa using h P hP
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem eq_zero_of_localization (r : R)
    (h : ∀ (J : Ideal R) (_ : J.IsMaximal), algebraMap R (Localization.AtPrime J) r = 0) :
    r = 0 :=
  Module.eq_zero_of_localization_maximal _ (fun _ _ ↦ Algebra.linearMap R _) r h


/-- An ideal is trivial if its localization at every maximal ideal is trivial. -/
theorem ideal_eq_bot_of_localization (I : Ideal R)
    (h : ∀ (J : Ideal R) (_ : J.IsMaximal),
      IsLocalization.coeSubmodule (Localization.AtPrime J) I = ⊥) :
    I = ⊥ :=
  bot_unique fun r hr ↦ eq_zero_of_localization r fun J hJ ↦ (h J hJ).le ⟨r, hr, rfl⟩


