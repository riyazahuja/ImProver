theorem iff_span_pair_isPrincipal :
    IsBezout R ↔ ∀ x y : R, (Ideal.span {x, y} : Ideal R).IsPrincipal := by
  classical
    constructor
    · intro H x y; infer_instance
    · intro H
      constructor
      apply Submodule.fg_induction
      · exact fun _ => ⟨⟨_, rfl⟩⟩
      · rintro _ _ ⟨⟨x, rfl⟩⟩ ⟨⟨y, rfl⟩⟩; rw [← Submodule.span_insert]; exact H _ _


theorem _root_.Function.Surjective.isBezout {S : Type v} [CommRing S] (f : R →+* S)
    (hf : Function.Surjective f) [IsBezout R] : IsBezout S := by
  /-
    R : Type u
    inst✝² : CommRing R
    S : Type v
    inst✝¹ : CommRing S
    f : RingHom R S
    hf : Function.Surjective ⇑f
    inst✝ : IsBezout R
    ⊢ IsBezout S
  -/
  rw [iff_span_pair_isPrincipal]
  /-
    R : Type u
    inst✝² : CommRing R
    S : Type v
    inst✝¹ : CommRing S
    f : RingHom R S
    hf : Function.Surjective ⇑f
    inst✝ : IsBezout R
    ⊢ ∀ (x y : S), Submodule.IsPrincipal (Ideal.span (Insert.insert x (Singleton.s …
  -/
  intro x y
  /-
    R : Type u
    inst✝² : CommRing R
    S : Type v
    inst✝¹ : CommRing S
    f : RingHom R S
    hf : Function.Surjective ⇑f
    inst✝ : IsBezout R
    x y : S
    ⊢ Submodule.IsPrincipal (Ideal.span (Insert.insert x (Singleton.singleton y)))
  -/
  obtain ⟨⟨x, rfl⟩, ⟨y, rfl⟩⟩ := hf x, hf y
  /-
    case intro.intro
    R : Type u
    inst✝² : CommRing R
    S : Type v
    inst✝¹ : CommRing S
    f : RingHom R S
    hf : Function.Surjective ⇑f
    inst✝ : IsBezout R
    x y : R
    ⊢ Submodule.IsPrincipal (Ideal.span (Insert.insert (f x) (Singleton.singleton  …
  -/
  use f (gcd x y)
  /-
    case h
    R : Type u
    inst✝² : CommRing R
    S : Type v
    inst✝¹ : CommRing S
    f : RingHom R S
    hf : Function.Surjective ⇑f
    inst✝ : IsBezout R
    x y : R
    ⊢ Eq (Ideal.span (Insert.insert (f x) (Singleton.singleton (f y)))) (Submodule …
  -/
  trans Ideal.map f (Ideal.span {gcd x y})
    /-
      R : Type u
      inst✝² : CommRing R
      S : Type v
      inst✝¹ : CommRing S
      f : RingHom R S
      hf : Function.Surjective ⇑f
      inst✝ : IsBezout R
      x y : R
      ⊢ Eq (Ideal.span (Insert.insert (f x) (Singleton.singleton (f y)))) (Ideal.map …
    -/
  · rw [span_gcd, Ideal.map_span, Set.image_insert_eq, Set.image_singleton]
    /-
      🎉 no goals
    -/
    /-
      R : Type u
      inst✝² : CommRing R
      S : Type v
      inst✝¹ : CommRing S
      f : RingHom R S
      hf : Function.Surjective ⇑f
      inst✝ : IsBezout R
      x y : R
      ⊢ Eq (Ideal.map f (Ideal.span (Singleton.singleton (IsBezout.gcd x y)))) (Subm …
    -/
  · rw [Ideal.map_span, Set.image_singleton]; rfl
                                              /-
                                                🎉 no goals
                                              -/


theorem TFAE [IsBezout R] [IsDomain R] :
    List.TFAE
    [IsNoetherianRing R, IsPrincipalIdealRing R, UniqueFactorizationMonoid R, WfDvdMonoid R] := by
  classical
    tfae_have 1 → 2
    | _ => inferInstance
    tfae_have 2 → 3
    | _ => inferInstance
    tfae_have 3 → 4
    | _ => inferInstance
    tfae_have 4 → 1
    | ⟨h⟩ => by
      rw [isNoetherianRing_iff, isNoetherian_iff_fg_wellFounded]
      refine ⟨RelEmbedding.wellFounded ?_ h⟩
      have : ∀ I : { J : Ideal R // J.FG }, ∃ x : R, (I : Ideal R) = Ideal.span {x} :=
        fun ⟨I, hI⟩ => (IsBezout.isPrincipal_of_FG I hI).1
      choose f hf using this
      exact
        { toFun := f
          inj' := fun x y e => by ext1; rw [hf, hf, e]
          map_rel_iff' := by
            dsimp
            intro a b
            rw [← Ideal.span_singleton_lt_span_singleton, ← hf, ← hf]
            rfl }
    tfae_finish


