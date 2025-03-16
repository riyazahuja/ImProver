/-- A ring homomorphism `f : R →+* S` is flat if `S` is flat as an `R` module. -/
@[algebraize Module.Flat]
def RingHom.Flat {R : Type u} {S : Type v} [CommRing R] [CommRing S] (f : R →+* S) : Prop :=
  letI : Algebra R S := f.toAlgebra
  Module.Flat R S


variable (R) in
/-- The identity of a ring is flat. -/
lemma id : RingHom.Flat (RingHom.id R) :=
  Module.Flat.self R


/-- Composition of flat ring homomorphisms is flat. -/
lemma comp {f : R →+* S} {g : S →+* T} (hf : f.Flat) (hg : g.Flat) : Flat (g.comp f) := by
  /-
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : CommRing T
    f : RingHom R S
    g : RingHom S T
    hf : f.Flat
    hg : g.Flat
    ⊢ (g.comp f).Flat
  -/
  algebraize [f, g, (g.comp f)]
  /-
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : CommRing T
    f : RingHom R S
    g : RingHom S T
    hf : f.Flat
    hg : g.Flat
    algInst✝² : Algebra R S := f.toAlgebra
    algInst✝¹ : Algebra S T := g.toAlgebra
    algInst✝ : Algebra R T := (g.comp f).toAlgebra
    scalarTowerInst✝ : IsScalarTower R S T := IsScalarTower.of_algebraMap_eq' (Eq. …
    algebraizeInst✝¹ : Module.Flat R S
    algebraizeInst✝ : Module.Flat S T
    ⊢ (g.comp f).Flat
  -/
  exact Module.Flat.trans R S T
  /-
    🎉 no goals
  -/


/-- Bijective ring maps are flat. -/
lemma of_bijective {f : R →+* S} (hf : Function.Bijective f) : Flat f := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    hf : Function.Bijective ⇑f
    ⊢ f.Flat
  -/
  algebraize [f]
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    hf : Function.Bijective ⇑f
    algInst✝ : Algebra R S := f.toAlgebra
    ⊢ f.Flat
  -/
  exact Module.Flat.of_linearEquiv R R S (LinearEquiv.ofBijective (Algebra.linearMap R S) hf).symm
  /-
    🎉 no goals
  -/


lemma containsIdentities : ContainsIdentities Flat := id


lemma stableUnderComposition : StableUnderComposition Flat := by
  /-
    ⊢ RingHom.StableUnderComposition fun {R S} [CommRing R] [CommRing S] => RingHo …
  -/
  introv R hf hg
  /-
    R S T : Type u_4
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : CommRing T
    f : RingHom R S
    g : RingHom S T
    hf : f.Flat
    hg : g.Flat
    ⊢ (g.comp f).Flat
  -/
  exact hf.comp hg
  /-
    🎉 no goals
  -/


lemma respectsIso : RespectsIso Flat := by
  /-
    ⊢ RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => RingHom.Flat
  -/
  apply stableUnderComposition.respectsIso
  /-
    ⊢ ∀ {R S : Type u_4} [inst : CommRing R] [inst_1 : CommRing S] (e : RingEquiv  …
  -/
  introv
  /-
    R S : Type u_4
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    e : RingEquiv R S
    ⊢ e.toRingHom.Flat
  -/
  exact of_bijective e.bijective
  /-
    🎉 no goals
  -/


lemma isStableUnderBaseChange : IsStableUnderBaseChange Flat := by
  /-
    ⊢ RingHom.IsStableUnderBaseChange fun {R S} [CommRing R] [CommRing S] => RingH …
  -/
  apply IsStableUnderBaseChange.mk _ respectsIso
  /-
    ⊢ ∀ ⦃R S T : Type u_4⦄ [inst : CommRing R] [inst_1 : CommRing S] [inst_2 : Com …
  -/
  introv h
  replace h : Module.Flat R T := by
    rw [RingHom.Flat] at h; convert h; ext; simp_rw [Algebra.smul_def]; rfl
  suffices Module.Flat S (S ⊗[R] T) by
    rw [RingHom.Flat]; convert this; congr; ext; simp_rw [Algebra.smul_def]; rfl
  /-
    R S T : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : CommRing T
    inst✝¹ : Algebra R S
    inst✝ : Algebra R T
    h : Module.Flat R T
    ⊢ Module.Flat S (TensorProduct R S T)
  -/
  exact inferInstance
  /-
    🎉 no goals
  -/


lemma holdsForLocalizationAway : HoldsForLocalizationAway Flat := by
  /-
    ⊢ RingHom.HoldsForLocalizationAway fun {R S} [CommRing R] [CommRing S] => Ring …
  -/
  introv R h
  suffices Module.Flat R S by
    rw [RingHom.Flat]; convert this; ext; simp_rw [Algebra.smul_def]; rfl
  /-
    R S : Type u_4
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    r : R
    h : IsLocalization.Away r S
    ⊢ Module.Flat R S
  -/
  exact IsLocalization.flat _ (Submonoid.powers r)
  /-
    🎉 no goals
  -/


lemma ofLocalizationSpanTarget : OfLocalizationSpanTarget Flat := by
  /-
    ⊢ RingHom.OfLocalizationSpanTarget fun {R S} [CommRing R] [CommRing S] => Ring …
  -/
  introv R hsp h
  /-
    R S : Type u_4
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set S
    hsp : Eq (Ideal.span s) Top.top
    h : ∀ (r : ↑s), (fun {R S} [CommRing R] [CommRing S] => RingHom.Flat) ((algebr …
    ⊢ f.Flat
  -/
  algebraize_only [f]
  refine Module.flat_of_isLocalized_span _ _ s hsp _
    (fun r ↦ Algebra.linearMap S <| Localization.Away r.1) ?_
  /-
    R S : Type u_4
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set S
    hsp : Eq (Ideal.span s) Top.top
    h : ∀ (r : ↑s), (fun {R S} [CommRing R] [CommRing S] => RingHom.Flat) ((algebr …
    algInst✝ : Algebra R S := f.toAlgebra
    ⊢ ∀ (r : ↑s), Module.Flat R (Localization.Away ↑r)
  -/
  dsimp only [RingHom.Flat] at h
  /-
    R S : Type u_4
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set S
    hsp : Eq (Ideal.span s) Top.top
    h : ∀ (r : ↑s), Module.Flat R (Localization.Away ↑r)
    algInst✝ : Algebra R S := f.toAlgebra
    ⊢ ∀ (r : ↑s), Module.Flat R (Localization.Away ↑r)
  -/
  convert h; ext
  /-
    case h.h.e'_5.smul.h.h
    R S : Type u_4
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set S
    hsp : Eq (Ideal.span s) Top.top
    h : ∀ (r : ↑s), Module.Flat R (Localization.Away ↑r)
    algInst✝ : Algebra R S := f.toAlgebra
    a✝ : ↑s
    x✝¹ : R
    x✝ : OreLocalization (Submonoid.powers ↑a✝) S
    ⊢ Eq (SMul.smul x✝¹ x✝) (SMul.smul x✝¹ x✝)
  -/
  apply Algebra.smul_def
  /-
    🎉 no goals
  -/


/-- Flat is a local property of ring homomorphisms. -/
lemma propertyIsLocal : PropertyIsLocal Flat where
  localizationAwayPreserves := isStableUnderBaseChange.localizationPreserves.away
  ofLocalizationSpanTarget := ofLocalizationSpanTarget
  ofLocalizationSpan := ofLocalizationSpanTarget.ofLocalizationSpan
    (stableUnderComposition.stableUnderCompositionWithLocalizationAway
      holdsForLocalizationAway).left
  StableUnderCompositionWithLocalizationAwayTarget :=
    (stableUnderComposition.stableUnderCompositionWithLocalizationAway
      holdsForLocalizationAway).right


