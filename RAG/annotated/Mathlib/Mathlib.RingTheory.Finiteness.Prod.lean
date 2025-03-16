theorem FG.prod {sb : Submodule R M} {sc : Submodule R P} (hsb : sb.FG) (hsc : sc.FG) :
    (sb.prod sc).FG :=
  let ⟨tb, htb⟩ := fg_def.1 hsb
  let ⟨tc, htc⟩ := fg_def.1 hsc
  fg_def.2
    ⟨LinearMap.inl R M P '' tb ∪ LinearMap.inr R M P '' tc, (htb.1.image _).union (htc.1.image _),
         /-
           R : Type u_1
           M : Type u_2
           inst✝⁴ : Semiring R
           inst✝³ : AddCommMonoid M
           inst✝² : Module R M
           P : Type u_3
           inst✝¹ : AddCommMonoid P
           inst✝ : Module R P
           sb : Submodule R M
           sc : Submodule R P
           hsb : sb.FG
           hsc : sc.FG
           tb : Set M
           htb : And tb.Finite (Eq (Submodule.span R tb) sb)
           tc : Set P
           htc : And tc.Finite (Eq (Submodule.span R tc) sc)
           ⊢ Eq (Submodule.span R (Union.union (Set.image (⇑(LinearMap.inl R M P)) tb) (S …
         -/
      by rw [LinearMap.span_inl_union_inr, htb.2, htc.2]⟩
         /-
           🎉 no goals
         -/


instance prod [hM : Module.Finite R M] [hN : Module.Finite R N] : Module.Finite R (M × N) :=
  ⟨by
    /-
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : AddCommMonoid N
      inst✝ : Module R N
      hM : Module.Finite R M
      hN : Module.Finite R N
      ⊢ Top.top.FG
    -/
    rw [← Submodule.prod_top]
    /-
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : AddCommMonoid N
      inst✝ : Module R N
      hM : Module.Finite R M
      hN : Module.Finite R N
      ⊢ (Top.top.prod Top.top).FG
    -/
    exact hM.1.prod hN.1⟩
    /-
      🎉 no goals
    -/


