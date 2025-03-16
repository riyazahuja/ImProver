lemma _root_.RingHom.injective_stableUnderComposition :
    RingHom.StableUnderComposition (fun f ↦ Function.Injective f) := by
  /-
    ⊢ RingHom.StableUnderComposition fun {R S} [CommRing R] [CommRing S] f => Func …
  -/
  intro R S T _ _ _ f g hf hg
  /-
    R S T : Type u_1
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : CommRing T
    f : RingHom R S
    g : RingHom S T
    hf : Function.Injective ⇑f
    hg : Function.Injective ⇑g
    ⊢ Function.Injective ⇑(g.comp f)
  -/
  simp only [RingHom.coe_comp]
  /-
    R S T : Type u_1
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : CommRing T
    f : RingHom R S
    g : RingHom S T
    hf : Function.Injective ⇑f
    hg : Function.Injective ⇑g
    ⊢ Function.Injective (Function.comp ⇑g ⇑f)
  -/
  exact Function.Injective.comp hg hf
  /-
    🎉 no goals
  -/


lemma _root_.RingHom.injective_respectsIso :
    RingHom.RespectsIso (fun f ↦ Function.Injective f) := by
  /-
    ⊢ RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] f => Function.Inject …
  -/
  apply RingHom.injective_stableUnderComposition.respectsIso
  /-
    ⊢ ∀ {R S : Type u_1} [inst : CommRing R] [inst_1 : CommRing S] (e : RingEquiv  …
  -/
  intro R S _ _ e
  /-
    R S : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    e : RingEquiv R S
    ⊢ Function.Injective ⇑e.toRingHom
  -/
  exact e.bijective.injective
  /-
    🎉 no goals
  -/

