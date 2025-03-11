instance pi (ι : Type u) [hi : Nonempty ι] (R : Type v) [Semiring R] (p : ℕ) [CharP R p] :
    CharP (ι → R) p :=
  ⟨fun x =>
    let ⟨i⟩ := hi
    Iff.symm <|
      (CharP.cast_eq_zero_iff R p x).symm.trans
        ⟨fun h =>
          funext fun j =>
                                                                   /-
                                                                     ι : Type u
                                                                     hi : Nonempty ι
                                                                     R : Type v
                                                                     inst✝¹ : Semiring R
                                                                     p : Nat
                                                                     inst✝ : CharP R p
                                                                     x : Nat
                                                                     i : ι
                                                                     h : Eq (↑x) 0
                                                                     j : ι
                                                                     ⊢ Eq ((Pi.evalRingHom (fun x => R) j) ↑x) 0
                                                                   -/
            show Pi.evalRingHom (fun _ => R) j (↑x : ι → R) = 0 by rw [map_natCast, h],
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
                                                                          /-
                                                                            ι : Type u
                                                                            hi : Nonempty ι
                                                                            R : Type v
                                                                            inst✝¹ : Semiring R
                                                                            p : Nat
                                                                            inst✝ : CharP R p
                                                                            x : Nat
                                                                            i : ι
                                                                            h : Eq (↑x) 0
                                                                            ⊢ Eq ((Pi.evalRingHom (fun x => R) i) ↑x) 0
                                                                          -/
          fun h => map_natCast (Pi.evalRingHom (fun _ : ι => R) i) x ▸ by rw [h, RingHom.map_zero]⟩⟩
                                                                          /-
                                                                            🎉 no goals
                                                                          -/

-- diamonds

instance pi' (ι : Type u) [Nonempty ι] (R : Type v) [CommRing R] (p : ℕ) [CharP R p] :
    CharP (ι → R) p :=
  CharP.pi ι R p


