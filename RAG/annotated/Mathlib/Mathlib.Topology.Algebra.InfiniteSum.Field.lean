nonrec theorem HasProd.norm (hfx : HasProd f x) : HasProd (‖f ·‖) ‖x‖ := by
  /-
    α : Type u_1
    E : Type u_2
    inst✝ : NormedField E
    f : α → E
    x : E
    hfx : HasProd f x
    ⊢ HasProd (fun x => Norm.norm (f x)) (Norm.norm x)
  -/
  simp only [HasProd, ← norm_prod]
  /-
    α : Type u_1
    E : Type u_2
    inst✝ : NormedField E
    f : α → E
    x : E
    hfx : HasProd f x
    ⊢ Filter.Tendsto (fun s => Norm.norm (s.prod fun b => f b)) Filter.atTop (nhds …
  -/
  exact hfx.norm
  /-
    🎉 no goals
  -/


theorem Multipliable.norm (hf : Multipliable f) : Multipliable (‖f ·‖) :=
  let ⟨x, hx⟩ := hf; ⟨‖x‖, hx.norm⟩


theorem norm_tprod (hf : Multipliable f) : ‖∏' i, f i‖ = ∏' i, ‖f i‖ :=
  hf.hasProd.norm.tprod_eq.symm


