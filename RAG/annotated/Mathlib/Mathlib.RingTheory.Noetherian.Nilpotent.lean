theorem IsNoetherianRing.isNilpotent_nilradical (R : Type*) [CommRing R] [IsNoetherianRing R] :
    IsNilpotent (nilradical R) := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsNoetherianRing R
    ⊢ IsNilpotent (nilradical R)
  -/
  obtain ⟨n, hn⟩ := Ideal.exists_radical_pow_le_of_fg (⊥ : Ideal R) (IsNoetherian.noetherian _)
  /-
    case intro
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsNoetherianRing R
    n : Nat
    hn : LE.le (HPow.hPow Bot.bot.radical n) Bot.bot
    ⊢ IsNilpotent (nilradical R)
  -/
  exact ⟨n, eq_bot_iff.mpr hn⟩
  /-
    🎉 no goals
  -/

