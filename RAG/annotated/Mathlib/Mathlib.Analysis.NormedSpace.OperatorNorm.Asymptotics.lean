theorem isBigOWith_id : IsBigOWith ‖f‖ l f fun x => x :=
  isBigOWith_of_le' _ f.le_opNorm


theorem isBigO_id : f =O[l] fun x => x :=
  (f.isBigOWith_id l).isBigO


theorem isBigOWith_comp [RingHomIsometric σ₂₃] {α : Type*} (g : F →SL[σ₂₃] G) (f : α → F)
    (l : Filter α) : IsBigOWith ‖g‖ l (fun x' => g (f x')) f :=
  (g.isBigOWith_id ⊤).comp_tendsto le_top


theorem isBigO_comp [RingHomIsometric σ₂₃] {α : Type*} (g : F →SL[σ₂₃] G) (f : α → F)
    (l : Filter α) : (fun x' => g (f x')) =O[l] f :=
  (g.isBigOWith_comp f l).isBigO


theorem isBigOWith_sub (x : E) :
    IsBigOWith ‖f‖ l (fun x' => f (x' - x)) fun x' => x' - x :=
  f.isBigOWith_comp _ l


theorem isBigO_sub (x : E) :
    (fun x' => f (x' - x)) =O[l] fun x' => x' - x :=
  f.isBigO_comp _ l


theorem isBigO_comp {α : Type*} (f : α → E) (l : Filter α) : (fun x' => e (f x')) =O[l] f :=
  (e : E →SL[σ₁₂] F).isBigO_comp f l


theorem isBigO_sub (l : Filter E) (x : E) : (fun x' => e (x' - x)) =O[l] fun x' => x' - x :=
  (e : E →SL[σ₁₂] F).isBigO_sub l x


theorem isBigO_comp_rev {α : Type*} (f : α → E) (l : Filter α) : f =O[l] fun x' => e (f x') :=
  (e.symm.isBigO_comp _ l).congr_left fun _ => e.symm_apply_apply _


theorem isBigO_sub_rev (l : Filter E) (x : E) : (fun x' => x' - x) =O[l] fun x' => e (x' - x) :=
  e.isBigO_comp_rev _ _


