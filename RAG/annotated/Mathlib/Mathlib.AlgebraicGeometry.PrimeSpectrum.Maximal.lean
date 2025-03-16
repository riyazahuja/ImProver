theorem toPrimeSpectrum_range :
    Set.range (@toPrimeSpectrum R _) = { x | IsClosed ({x} : Set <| PrimeSpectrum R) } := by
  /-
    R : Type u
    inst✝ : CommRing R
    ⊢ Eq (Set.range MaximalSpectrum.toPrimeSpectrum) (setOf fun x => IsClosed (Sin …
  -/
  simp only [isClosed_singleton_iff_isMaximal]
  /-
    R : Type u
    inst✝ : CommRing R
    ⊢ Eq (Set.range MaximalSpectrum.toPrimeSpectrum) (setOf fun x => x.asIdeal.IsM …
  -/
  ext ⟨x, _⟩
  /-
    case h.mk
    R : Type u
    inst✝ : CommRing R
    x : Ideal R
    isPrime✝ : x.IsPrime
    ⊢ Iff (Membership.mem (Set.range MaximalSpectrum.toPrimeSpectrum) { asIdeal := …
  -/
  exact ⟨fun ⟨y, hy⟩ => hy ▸ y.IsMaximal, fun hx => ⟨⟨x, hx⟩, rfl⟩⟩
  /-
    🎉 no goals
  -/


/-- The Zariski topology on the maximal spectrum of a commutative ring is defined as the subspace
topology induced by the natural inclusion into the prime spectrum. -/
instance zariskiTopology : TopologicalSpace <| MaximalSpectrum R :=
  PrimeSpectrum.zariskiTopology.induced toPrimeSpectrum


instance : T1Space <| MaximalSpectrum R :=
  ⟨fun x => isClosed_induced_iff.mpr
    ⟨{toPrimeSpectrum x}, (isClosed_singleton_iff_isMaximal _).mpr x.IsMaximal, by
      /-
        R : Type u
        inst✝ : CommRing R
        x : MaximalSpectrum R
        ⊢ Eq (Set.preimage MaximalSpectrum.toPrimeSpectrum (Singleton.singleton x.toPr …
      -/
      simpa only [← image_singleton] using preimage_image_eq {x} toPrimeSpectrum_injective⟩⟩
      /-
        🎉 no goals
      -/


theorem toPrimeSpectrum_continuous : Continuous <| @toPrimeSpectrum R _ :=
  continuous_induced_dom


